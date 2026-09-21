# ------------------------------------------------------------------------
# ADTI-Net (base model): video object detection.
#
# Reference:
#   "ADTI-Net: Alternating Decoupled Transformer Imitation Network for
#    Video Object Detection" (IEEE TIP).
#
# Pipeline (Fig. 3):
#   backbone (shared) -> tokenize -> vanilla transformer encoder
#   -> ADTD (ST-DTD / TS-DTD branches + DAFC, see adti_transformer.py)
#   -> TFIL (training only) + shared FFN detection head
#   -> detection results of the frames.
#
# Implementation details (Sec. IV-C):
#   * 72 object queries per frame (ResNet-101, ImageNet VID);
#   * 4 attention heads;
#   * TFIL: textual embedding dim 128, temperature tau=0.5, adaptive
#     threshold alpha = mean of the quality scores (Eq. 7/8), imitation
#     loss as in Eq. 9;
#   * training input: 1 current frame + 4 randomly sampled support frames;
#     inference: N=30 frames per sequence (see configs/).
#
# The criterion is self-contained (does not import the compiled CUDA ops),
# so the base ADTI-Net also trains/evaluates without `models/ops` compiled.
# ------------------------------------------------------------------------

import json
import math
import os

import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc_multi import (NestedTensor, nested_tensor_from_tensor_list,
                             accuracy, get_world_size,
                             is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import sigmoid_focal_loss
from .adti_transformer import build_adti_transformer, MLP
from .text_encoder import TextEncoder


# Fallback class-name list of the 30 ILSVRC (ImageNet VID) categories,
# ordered by category id (1..30). The class names are normally read from
# the annotation json at build time (see `load_vid_class_names`).
IMAGENET_VID_CLASSES = [
    'airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
    'cattle', 'dog', 'domestic_cat', 'dragonfly', 'elephant', 'fox',
    'giant_panda', 'hamster', 'horse', 'leopard', 'lion', 'lizard',
    'monkey', 'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
    'squirrel', 'tiger', 'train', 'turtle', 'whale',
]


def load_vid_class_names(vid_path):
    """Read the class names (sorted by category id) from the dataset
    annotation json, so that the textual features are aligned with the
    category ids used by the dataloader."""
    candidates = ['imagenet_vid_val.json',
                  'imagenet_vid_train_joint_30.json']
    for name in candidates:
        path = os.path.join(vid_path, 'annotations', name)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    cats = json.load(f)['categories']
                cats = sorted(cats, key=lambda c: c['id'])
                return [c['name'] for c in cats]
            except Exception as e:
                print('[ADTI-Net] failed to read categories from {}: {}'
                      .format(path, e))
    print('[ADTI-Net] WARNING: annotation json not found under {}; using '
          'the default ImageNet VID class list for the text encoder. '
          'Please verify it against your dataset.'.format(vid_path))
    return None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class ADTINet(nn.Module):
    """ADTI-Net base detector.

    The input `samples` contains all F frames of one clip stacked along the
    batch dimension (F = 1 current frame + num_ref_frames support frames,
    following the video dataloader; the current frame is at index 0).
    """

    def __init__(self, backbone, transformer, num_classes=31,
                 num_queries=72, aux_loss=True):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        # single feature level (C5), as in the Deformable DETR baseline
        self.input_proj = nn.ModuleList([
            nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
            nn.GroupNorm(32, hidden_dim),
        ])
        self.backbone = backbone
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0.)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
        nn.init.constant_(self.input_proj[0].weight, 0.)
        nn.init.constant_(self.input_proj[0].bias, 0.)

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)

        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None
        # single feature level
        src = self.input_proj[0](src)
        proj_mask = mask
        pos_embed = pos[-1]

        num_frames = src.shape[0]
        src_flatten = src.flatten(2).transpose(1, 2)      # (F, hw, C)
        mask_flatten = proj_mask.flatten(1)               # (F, hw)
        pos_flatten = pos_embed.flatten(2).transpose(1, 2)
        spatial_shapes = (proj_mask.shape[-2], proj_mask.shape[-1])

        # object queries (shared by all frames)
        query_embed = self.query_embed.weight              # (nq, 2C)
        query_pos, tgt = torch.split(
            query_embed, self.transformer.d_model, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(num_frames, -1, -1)
        tgt = tgt.unsqueeze(0).expand(num_frames, -1, -1)

        coupled, final_ref, intermediates, memory = self.transformer(
            src_flatten, pos_flatten, mask_flatten, tgt, query_pos,
            spatial_shapes)

        # final predictions (all frames; frame 0 is the current frame)
        outputs_class = self.class_embed(coupled)          # (F, nq, K)
        tmp = self.bbox_embed(coupled)
        tmp[..., :2] = tmp[..., :2] + inverse_sigmoid(final_ref)
        outputs_coord = tmp.sigmoid()

        out = {
            'pred_logits': outputs_class[0:1],             # (1, nq, K)
            'pred_boxes': outputs_coord[0:1],              # (1, nq, 4)
            'pred_logits_all': outputs_class,
            'pred_boxes_all': outputs_coord,
            # coupled features of the current frame for the TFIL loss
            'coupled_feats': coupled[0:1],
        }
        if self.aux_loss:
            aux_outputs = []
            for layer_out, layer_ref, layer in intermediates:
                a_cls = self.class_embed(layer_out)
                a_tmp = layer.bbox_embed(layer_out)
                a_tmp[..., :2] = a_tmp[..., :2] + inverse_sigmoid(layer_ref)
                aux_outputs.append({
                    'pred_logits': a_cls[0:1],
                    'pred_boxes': a_tmp.sigmoid()[0:1],
                })
            out['aux_outputs'] = aux_outputs
        return out


# ---------------------------------------------------------------------------
# TFIL: feature quality indicator (Eq. 7 / Eq. 8)
# ---------------------------------------------------------------------------
class FeatureQualityIndicator(nn.Module):
    """Q(c_i, y_j) = Sigmoid(Upsilon^T rho(Pi Concat(c_i, y_j)))."""

    def __init__(self, d_model=256, text_dim=128):
        super().__init__()
        self.proj = nn.Linear(d_model + text_dim, d_model)   # Pi
        self.act = nn.ReLU()                                 # rho
        self.score = nn.Linear(d_model, 1)                   # Upsilon

    def forward(self, feats, text_feats):
        """
        :param feats       (M, C) coupled features
        :param text_feats  (K, T) textual features
        :return (M, K) quality scores in (0, 1)
        """
        M, C = feats.shape
        K, T = text_feats.shape
        f = feats[:, None, :].expand(M, K, C)
        t = text_feats[None, :, :].expand(M, K, T)
        cat = torch.cat([f, t], dim=-1)
        return self.score(self.act(self.proj(cat))).squeeze(-1).sigmoid()


# ---------------------------------------------------------------------------
# Criterion (detection losses of Deformable DETR + imitation loss of TFIL)
# ---------------------------------------------------------------------------
class ADTISetCriterion(nn.Module):
    """Detection loss (focal + L1 + GIoU with Hungarian matching, identical
    to ADTI-Net / Deformable DETR) plus the imitation loss of TFIL (Eq. 9)."""

    def __init__(self, num_classes, matcher, weight_dict, losses,
                 focal_alpha=0.25, text_encoder=None,
                 quality_indicator=None, film_embed=None, film_tau=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        # TFIL modules
        self.text_encoder = text_encoder
        self.quality_indicator = quality_indicator
        self.film_embed = film_embed
        self.film_tau = film_tau

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Focal classification loss."""
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout,
            device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot,
                                     self.num_classes,
                                     alpha=self.focal_alpha, gamma=2,
                                     reduction='none') * src_logits.shape[1]
        losses = {'loss_ce': loss_ce.mean()}
        if log:
            losses['class_error'] = \
                100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets],
                                      device=device)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # ------------------------------------------------------------------
    # TFIL imitation loss (Eq. 7 - Eq. 9)
    # ------------------------------------------------------------------
    def loss_film(self, feats, indices, targets):
        """Text-driven feature imitation loss over the matched coupled
        features of the current frame."""
        idx = self._get_src_permutation_idx(indices)
        src_feats = feats[idx]                       # (M, C)
        if src_feats.shape[0] == 0:
            return feats.sum() * 0
        labels = torch.cat([t['labels'][J]
                            for t, (_, J) in zip(targets, indices)])  # (M,)

        text_feats = self.text_encoder()             # (K, T)
        # quality scores between every coupled feature and every class
        quality = self.quality_indicator(src_feats, text_feats)  # (M, K)
        # adaptive threshold (Eq. 7)
        alpha = quality.mean()
        max_q = quality.max(dim=1).values
        exemplar_mask = max_q >= alpha
        non_exemplar_mask = ~exemplar_mask
        if exemplar_mask.sum() == 0 or non_exemplar_mask.sum() == 0:
            return feats.sum() * 0

        # embed into the imitation space (dim 128)
        o = self.film_embed(src_feats[non_exemplar_mask])   # (Mn, T)
        r = self.film_embed(src_feats[exemplar_mask])       # (Me, T)
        lab_o = labels[non_exemplar_mask]
        lab_r = labels[exemplar_mask]

        # cosine similarity / tau (Eq. 9)
        sim = torch.cosine_similarity(
            o.unsqueeze(1), r.unsqueeze(0), dim=-1) / self.film_tau  # (Mn, Me)
        pos = (lab_o.unsqueeze(1) == lab_r.unsqueeze(0)).to(sim.dtype)
        has_pos = pos.sum(dim=1) > 0
        if not has_pos.any():
            return feats.sum() * 0
        sim = sim[has_pos]
        pos = pos[has_pos]

        log_denom = torch.logsumexp(sim, dim=1, keepdim=True)
        per_pair = sim - log_denom
        row_loss = (per_pair * pos).sum(dim=1) / pos.sum(dim=1).clamp(min=1)
        return -row_loss.mean()

    # ------------------------------------------------------------------
    def forward(self, outputs, targets):
        feats = outputs.get('coupled_feats', None)
        outputs_without_aux = {
            k: v for k, v in outputs.items()
            if k not in ('aux_outputs', 'enc_outputs', 'coupled_feats')}

        # Hungarian matching (also needed by the imitation loss)
        indices = self.matcher(outputs_without_aux, targets)

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            if loss == 'labels':
                losses.update(
                    self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss == 'boxes':
                losses.update(
                    self.loss_boxes(outputs, targets, indices, num_boxes))
            elif loss == 'cardinality':
                losses.update(self.loss_cardinality(
                    outputs, targets, indices, num_boxes))
            else:
                raise ValueError('unsupported loss {}'.format(loss))

        # imitation loss of TFIL (training only)
        if feats is not None and self.text_encoder is not None:
            losses['loss_film'] = self.loss_film(feats, indices, targets)

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                aux_indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'labels':
                        l_dict = self.loss_labels(
                            aux_outputs, targets, aux_indices, num_boxes,
                            log=False)
                    elif loss == 'boxes':
                        l_dict = self.loss_boxes(
                            aux_outputs, targets, aux_indices, num_boxes)
                    elif loss == 'cardinality':
                        l_dict = self.loss_cardinality(
                            aux_outputs, targets, aux_indices, num_boxes)
                    else:
                        raise ValueError('unsupported loss {}'.format(loss))
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


# ---------------------------------------------------------------------------
# Post-processing (identical to ADTI-Net / Deformable DETR)
# ---------------------------------------------------------------------------
class PostProcess(nn.Module):
    """Converts model output into the COCO api format."""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.split(1, 2)
        scale_fct = torch.cat(
            [img_w, img_h, img_w, img_h],
            dim=-1).repeat(1, 1, 2).to(boxes.device)
        boxes = boxes * scale_fct

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]
        return results


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(args):
    # ADTI-Net uses a single feature level (C5), like the Deformable DETR
    # baseline of the paper
    assert args.num_feature_levels == 1, \
        'ADTI-Net (base) is implemented with a single feature level.'

    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_adti_transformer(args)
    model = ADTINet(
        backbone,
        transformer,
        num_classes=31,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_bbox': args.bbox_loss_coef,
        'loss_giou': args.giou_loss_coef,
    }
    num_aux = transformer.adtd.num_layers_total
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(num_aux):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    # weight of the imitation loss (training only, not reported at eval)
    weight_dict['loss_film'] = args.film_loss_coef

    class_names = load_vid_class_names(args.vid_path) or IMAGENET_VID_CLASSES
    text_encoder = TextEncoder(
        class_names, embed_dim=args.film_dim, model_name=args.text_model)
    quality_indicator = FeatureQualityIndicator(
        args.hidden_dim, args.film_dim)
    film_embed = nn.Linear(args.hidden_dim, args.film_dim)

    losses = ['labels', 'boxes', 'cardinality']
    criterion = ADTISetCriterion(
        31, matcher=matcher, weight_dict=weight_dict, losses=losses,
        focal_alpha=args.focal_alpha, text_encoder=text_encoder,
        quality_indicator=quality_indicator, film_embed=film_embed,
        film_tau=args.film_tau)
    criterion.to(device)

    postprocessors = {'bbox': PostProcess()}

    return model, criterion, postprocessors
