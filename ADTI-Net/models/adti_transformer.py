# ------------------------------------------------------------------------
# ADTI-Net (base model): video object detection.
# ADTD: Alternating Decoupled Transformer Decoder.
#
# Reference:
#   "ADTI-Net: Alternating Decoupled Transformer Imitation Network for
#    Video Object Detection" (IEEE TIP).
#
# Structure (Fig. 3 / Fig. 4 of the paper):
#   token features -> vanilla transformer encoder (per frame, shared weights)
#   object queries -> ADTD
#     |- ST-DTD branch: S-DTD layers (spatial)  -> T-DTD layers (temporal)
#     |- TS-DTD branch: T-DTD layers (temporal) -> S-DTD layers (spatial)
#     |- DAFC: adaptive feature coupling of the two branch outputs
#   coupled features -> shared FFN detection head
#
# Each S-DTD layer = spatial mask self-attention (local window G)
#                  + spatial deformable attention (to its own frame)
#                  + FFN.
# Each T-DTD layer = temporal mask self-attention (local window G over all
#                    frames' queries)
#                  + temporal deformable attention (to all frames, Eq. 5)
#                  + FFN.
#
# Implementation details from the paper (Sec. IV-C):
#   * 2 spatially-decoupled + 3 temporally-decoupled decoder layers
#     in each branch;
#   * number of attention heads: 4;
#   * local window size G of the mask self-attention: 256.
#
# Note: the deformable attentions re-implement MSDeformAttn with a pure
# PyTorch fallback, so the model also runs when the CUDA ops of
# `models/ops` are not compiled (e.g. on Windows). When the compiled op is
# available it is used automatically.
# ------------------------------------------------------------------------

import copy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_

from util.misc import inverse_sigmoid

try:
    from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction
    _HAS_MSDA_CUDA = True
except Exception:  # pragma: no cover - ops not compiled
    MSDeformAttnFunction = None
    _HAS_MSDA_CUDA = False


def _is_power_of_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1) == 0)


def _ms_deform_attn_core_pytorch(value, value_spatial_shapes,
                                 sampling_locations, attention_weights):
    """Pure PyTorch reference implementation of multi-scale deformable
    attention (from Deformable DETR). Used when the CUDA op is missing."""
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([int(_) for _ in value_spatial_shapes[:, 0] *
                              value_spatial_shapes[:, 1]], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(
            N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(N_, M_ * Lq_, D_)
    return output.transpose(1, 2).contiguous()


def deform_attn_core(value, value_spatial_shapes, value_level_start_index,
                     sampling_locations, attention_weights, im2col_step=64):
    """Dispatch between the compiled CUDA op and the PyTorch fallback."""
    if _HAS_MSDA_CUDA and value.is_cuda:
        # the CUDA kernel requires batch % im2col_step == 0
        im2col_step = max(1, min(int(im2col_step), value.shape[0]))
        return MSDeformAttnFunction.apply(
            value, value_spatial_shapes, value_level_start_index,
            sampling_locations, attention_weights, im2col_step)
    return _ms_deform_attn_core_pytorch(
        value, value_spatial_shapes, sampling_locations, attention_weights)


def get_valid_ratio(mask):
    """Valid (non-padded) ratio of each frame's feature map (Deformable DETR)."""
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    if activation == 'gelu':
        return F.gelu
    if activation == 'glu':
        return F.glu
    raise RuntimeError(F'activation should be relu/gelu, not {activation}.')


# ---------------------------------------------------------------------------
# Vanilla transformer encoder (per-frame, weight sharing)
# ---------------------------------------------------------------------------
class VanillaTransformerEncoderLayer(nn.Module):
    """Standard transformer encoder layer applied to every frame
    independently (the frames form the batch dimension)."""

    def __init__(self, d_model, d_ffn, dropout, activation, nhead):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos, src_key_padding_mask=None):
        # src: [F, hw, C], pos: [F, hw, C], mask: [F, hw]
        q = k = src + pos
        src2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                              src.transpose(0, 1),
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = src2.transpose(0, 1)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class VanillaTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos, src_key_padding_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, pos, src_key_padding_mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ---------------------------------------------------------------------------
# Mask self-attention with a fixed-size local window (Eq. 1 / Eq. 4)
# ---------------------------------------------------------------------------
class WindowedSelfAttention(nn.Module):
    """Multi-head self-attention where each query only attends to the
    key features within a fixed G-size local window centered on it."""

    def __init__(self, d_model, nhead, dropout=0.1, window=256):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.window = window
        self._mask_cache = {}

    def _banded_mask(self, L, device):
        key = (L, str(device))
        if key not in self._mask_cache:
            half = self.window // 2
            idx = torch.arange(L, device=device)
            blocked = (idx[:, None] - idx[None, :]).abs() > half
            mask = torch.zeros(L, L, device=device)
            mask.masked_fill_(blocked, float('-inf'))
            self._mask_cache[key] = mask
        return self._mask_cache[key]

    def forward(self, query, key, value):
        # query/key/value: [L, B, C]
        L = query.shape[0]
        attn_mask = None
        if L > self.window:
            attn_mask = self._banded_mask(L, query.device)
        return self.attn(query, key, value, attn_mask=attn_mask)[0]


# ---------------------------------------------------------------------------
# Deformable attention (re-implementation of MSDeformAttn with fallback)
# ---------------------------------------------------------------------------
class DeformableAttention(nn.Module):
    """(Multi-scale) deformable attention with n_levels levels; identical
    math to MSDeformAttn of Deformable DETR, but runs with a pure PyTorch
    core when the compiled CUDA op is unavailable.

    `reference_points` must already be scaled by the per-level valid ratios
    (as done by the original Deformable DETR decoder)."""

    def __init__(self, d_model=256, n_levels=1, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got '
                             '{} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        if not _is_power_of_2(_d_per_head):
            import warnings
            warnings.warn('d_model/n_heads is not a power of 2; this is less '
                          'efficient for the CUDA implementation.')

        self.im2col_step = 64
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        import math
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(
            self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]) \
            .view(self.n_heads, 1, 1, 2).repeat(
                1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten,
                input_spatial_shapes, input_level_start_index,
                input_padding_mask=None):
        """
        :param query                   (N, Len_q, C)
        :param reference_points        (N, Len_q, n_levels, 2) in [0, 1],
                                       valid-ratio scaled, incl. padding area
        :param input_flatten           (N, sum(H_l*W_l), C)
        :param input_spatial_shapes    (n_levels, 2)
        :param input_level_start_index (n_levels,)
        :param input_padding_mask      (N, sum(H_l*W_l)), True for padding
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] *
                input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads,
                           self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [input_spatial_shapes[..., 1],
                 input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.n_points \
                * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, '
                             'but get {} instead.'.format(
                                 reference_points.shape[-1]))

        output = deform_attn_core(
            value, input_spatial_shapes, input_level_start_index,
            sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


# ---------------------------------------------------------------------------
# Temporal deformable attention (Eq. 5)
# ---------------------------------------------------------------------------
class TemporalDeformAttn(nn.Module):
    """Temporal multi-head deformable attention.

    The frames of the input sequence are treated as the `levels` of the
    deformable attention, i.e. every query samples from the feature maps of
    all N frames at its (frame-shared) normalized reference point plus
    learned offsets (Eq. 5; since all frames share the feature resolution,
    the normalized coordinates are identical across frames).

    The linear layers are built for a fixed maximum number of levels
    (`n_levels_max`, paper: N=30 at inference); when the actual number of
    frames F < n_levels_max during training, the levels are mapped onto the
    F frames by index rounding (i.e. each frame is sampled by several
    levels). F == n_levels_max gives an exact 1:1 mapping."""

    def __init__(self, d_model=256, n_heads=8, n_points=4, n_levels_max=30):
        super().__init__()
        self.n_levels_max = n_levels_max
        self.inner = DeformableAttention(d_model, n_levels_max, n_heads,
                                         n_points)

    def forward(self, query, reference_points, frame_src, frame_padding_mask,
                spatial_shape, valid_ratios):
        """
        :param query               (1, F*nq, C)  queries of all frames, flattened
        :param reference_points    (1, F*nq, 2)  normalized reference points
        :param frame_src           (1, F*hw, C)  encoded features of all frames
        :param frame_padding_mask  (1, F*hw) or None
        :param spatial_shape        (h, w)
        :param valid_ratios         (1, F, 2) per-frame valid ratios
        """
        device = query.device
        H, W = int(spatial_shape[0]), int(spatial_shape[1])
        hw = H * W
        F_ = valid_ratios.shape[1]
        Lmax = self.n_levels_max
        if F_ > Lmax:
            raise ValueError(
                'Number of frames ({}) exceeds n_levels_max ({}); increase '
                '--max_seq_frames.'.format(F_, Lmax))

        # level -> frame index mapping
        if F_ == Lmax:
            level_frame_idx = torch.arange(F_, device=device)
        else:
            level_frame_idx = torch.div(
                torch.arange(Lmax, device=device) * F_, Lmax,
                rounding_mode='floor').clamp(max=F_ - 1)

        # build the level-stacked value input [1, Lmax*hw, C]
        tok_idx = (level_frame_idx[:, None] * hw +
                   torch.arange(hw, device=device)[None, :]).reshape(-1)
        value_input = frame_src.index_select(1, tok_idx)
        if frame_padding_mask is not None:
            mask_input = frame_padding_mask.index_select(1, tok_idx)
        else:
            mask_input = None

        value_spatial_shapes = torch.as_tensor(
            [[H, W]] * Lmax, dtype=torch.long, device=device)
        value_level_start_index = torch.cat(
            (value_spatial_shapes.new_zeros((1,)),
             value_spatial_shapes.prod(1).cumsum(0)[:-1]))

        # per-level valid ratios (of the mapped frames)
        valid_ratios_levels = valid_ratios.index_select(1, level_frame_idx)
        # (1, Lq, Lmax, 2)
        reference_points_input = reference_points[:, :, None] * \
            valid_ratios_levels[:, None]

        return self.inner(query, reference_points_input, value_input,
                          value_spatial_shapes, value_level_start_index,
                          mask_input)


# ---------------------------------------------------------------------------
# S-DTD / T-DTD decoder layers
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Simple multi-layer perceptron (from DETR/Deformable DETR)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class _DTDLayer(nn.Module):
    """Common utilities of the S-DTD / T-DTD layers (FFN, refinement)."""

    def _init_common(self, d_model, d_ffn, dropout, activation):
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        # per-layer iterative box refinement head
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0.)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0.)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

    def _forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def _refine(self, tgt, reference_points):
        """Iterative reference point refinement (Deformable DETR)."""
        tmp = self.bbox_embed(tgt)
        new_reference = tmp[..., :2] + inverse_sigmoid(reference_points)
        new_reference = new_reference.sigmoid()
        return new_reference.detach()


class SpatialDTDLayer(_DTDLayer):
    """Spatially-decoupled transformer decoder layer (S-DTD).

    Spatial mask self-attention among the queries of each single frame
    (Eq. 1) + spatial deformable attention between the queries and the
    encoded features of the same frame (Eq. 3) + FFN."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 activation='relu', n_heads=8, n_points=4, window=256):
        super().__init__()
        # spatial multi-head mask self-attention (Eq. 1)
        self.self_attn = WindowedSelfAttention(d_model, n_heads, dropout,
                                               window)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # spatial multi-head deformable attention (Eq. 3)
        self.cross_attn = DeformableAttention(d_model, 1, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self._init_common(d_model, d_ffn, dropout, activation)

    def forward(self, tgt, query_pos, reference_points, src,
                spatial_shapes, level_start_index, valid_ratios,
                src_padding_mask=None):
        """
        :param tgt              (F, nq, C) queries of all frames
        :param reference_points (F, nq, 2)
        :param src              (F, hw, C) encoded features (frames = batch)
        """
        # spatial mask self-attention, within each frame
        q = k = tgt + query_pos
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1),
                              tgt.transpose(0, 1)).transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # spatial deformable attention to the same-frame features
        reference_points_input = reference_points[:, :, None] * \
            valid_ratios[:, None]                       # (F, nq, 1, 2)
        tgt2 = self.cross_attn(tgt + query_pos, reference_points_input, src,
                               spatial_shapes, level_start_index,
                               src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self._forward_ffn(tgt)

        new_reference = self._refine(tgt, reference_points)
        return tgt, new_reference


class TemporalDTDLayer(_DTDLayer):
    """Temporally-decoupled transformer decoder layer (T-DTD).

    Temporal mask self-attention among the queries of all frames (Eq. 4)
    + temporal deformable attention to the encoded features of all N frames
    (Eq. 5) + FFN."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 activation='relu', n_heads=8, n_points=4, window=256,
                 n_levels_max=30):
        super().__init__()
        # temporal multi-head mask self-attention (Eq. 4)
        self.self_attn = WindowedSelfAttention(d_model, n_heads, dropout,
                                               window)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # temporal multi-head deformable attention (Eq. 5)
        self.cross_attn = TemporalDeformAttn(d_model, n_heads, n_points,
                                             n_levels_max)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self._init_common(d_model, d_ffn, dropout, activation)

    def forward(self, tgt, query_pos, reference_points, frame_src,
                frame_padding_mask, spatial_shape, valid_ratios):
        """
        :param tgt              (F, nq, C)
        :param reference_points (F, nq, 2)
        :param frame_src        (1, F*hw, C)
        :param valid_ratios     (1, F, 2)
        """
        F_, nq, C = tgt.shape
        # temporal mask self-attention over the flattened query sequence
        q = tgt + query_pos                                   # (F, nq, C)
        q_flat = q.reshape(1, F_ * nq, C).transpose(0, 1)     # (F*nq, 1, C)
        v_flat = tgt.reshape(1, F_ * nq, C).transpose(0, 1)
        tgt2 = self.self_attn(q_flat, q_flat, v_flat)
        tgt2 = tgt2.transpose(0, 1).reshape(F_, nq, C)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # temporal deformable attention to all frames
        q2 = (tgt + query_pos).reshape(1, F_ * nq, C)
        ref_flat = reference_points.reshape(1, F_ * nq, 2)
        tgt2 = self.cross_attn(q2, ref_flat, frame_src, frame_padding_mask,
                               spatial_shape, valid_ratios)
        tgt2 = tgt2.reshape(F_, nq, C)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt = self._forward_ffn(tgt)

        new_reference = self._refine(tgt, reference_points)
        return tgt, new_reference


# ---------------------------------------------------------------------------
# DAFC (dual-branch adaptive feature coupling, Eq. 6)
# ---------------------------------------------------------------------------
class DAFC(nn.Module):
    """Adaptively couple the outputs of the ST-DTD branch (X) and the
    TS-DTD branch (Y):

        Theta = [theta_ST, theta_TS] = Conv1x1(Concat(X, Y)),
        C = theta_ST * X + theta_TS * Y  (element-wise).

    The 1x1 1D convolution is realized as a Linear layer over the channel
    dimension, producing two scalar gates per token."""

    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Linear(d_model * 2, 2)

    def forward(self, x, y):
        w = self.gate(torch.cat([x, y], dim=-1)).sigmoid()
        theta_st, theta_ts = w[..., 0:1], w[..., 1:2]
        return theta_st * x + theta_ts * y


# ---------------------------------------------------------------------------
# ADTD branches and the full ADTD module
# ---------------------------------------------------------------------------
class ADTDBranch(nn.Module):
    """One branch of the ADTD (Fig. 4(a)).

    order='st': n_s spatially-decoupled layers followed by n_t
                temporally-decoupled layers (ST-DTD);
    order='ts': n_t temporally-decoupled layers followed by n_s
                spatially-decoupled layers (TS-DTD)."""

    def __init__(self, order, d_model, d_ffn, dropout, activation, n_heads,
                 n_points, window, n_s_layers, n_t_layers, n_levels_max):
        super().__init__()
        assert order in ('st', 'ts')
        self.order = order
        s_layer = SpatialDTDLayer(d_model, d_ffn, dropout, activation,
                                  n_heads, n_points, window)
        t_layer = TemporalDTDLayer(d_model, d_ffn, dropout, activation,
                                   n_heads, n_points, window, n_levels_max)
        s_layers = _get_clones(s_layer, n_s_layers)
        t_layers = _get_clones(t_layer, n_t_layers)
        self.layers = nn.ModuleList(
            s_layers + t_layers if order == 'st' else t_layers + s_layers)

    def forward(self, tgt, query_pos, reference_points, spatial_ctx,
                temporal_ctx):
        intermediates = []
        for layer in self.layers:
            if isinstance(layer, SpatialDTDLayer):
                tgt, reference_points = layer(
                    tgt, query_pos, reference_points, **spatial_ctx)
            else:
                tgt, reference_points = layer(
                    tgt, query_pos, reference_points, **temporal_ctx)
            intermediates.append((tgt, reference_points, layer))
        return tgt, reference_points, intermediates


class ADTD(nn.Module):
    """Alternating Decoupled Transformer Decoder: ST-DTD branch + TS-DTD
    branch in parallel, coupled by DAFC."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1,
                 activation='relu', n_heads=8, n_points=4, window=256,
                 n_s_layers=2, n_t_layers=3, n_levels_max=30):
        super().__init__()
        self.n_s_layers = n_s_layers
        self.n_t_layers = n_t_layers
        self.st_branch = ADTDBranch('st', d_model, d_ffn, dropout,
                                    activation, n_heads, n_points, window,
                                    n_s_layers, n_t_layers, n_levels_max)
        self.ts_branch = ADTDBranch('ts', d_model, d_ffn, dropout,
                                    activation, n_heads, n_points, window,
                                    n_s_layers, n_t_layers, n_levels_max)
        self.dafc = DAFC(d_model)

    @property
    def num_layers_total(self):
        """Total number of decoder layers (both branches), used for the
        auxiliary losses."""
        return 2 * (self.n_s_layers + self.n_t_layers)

    def forward(self, tgt, query_pos, reference_points, memory,
                mask_flatten, spatial_shapes):
        """
        :param tgt              (F, nq, C) initial content queries
        :param query_pos        (F, nq, C) positional queries
        :param reference_points (F, nq, 2)
        :param memory           (F, hw, C) encoded features of all frames
        :param mask_flatten     (F, hw) padding mask
        :param spatial_shapes   (h, w) of the single feature level
        """
        F_ = memory.shape[0]
        H, W = int(spatial_shapes[0]), int(spatial_shapes[1])
        device = memory.device

        # context for the spatial layers (frames as batch, single level)
        valid_ratios_frames = get_valid_ratio(
            mask_flatten.reshape(F_, H, W))                   # (F, 2)
        spatial_ctx = dict(
            src=memory,
            spatial_shapes=torch.as_tensor([[H, W]], dtype=torch.long,
                                           device=device),
            level_start_index=torch.zeros(1, dtype=torch.long, device=device),
            valid_ratios=valid_ratios_frames[:, None, :],   # (F, 1, 2)
            src_padding_mask=mask_flatten,
        )

        # context for the temporal layers (frames as levels)
        frame_src = memory.reshape(1, F_ * H * W, memory.shape[-1])
        frame_padding_mask = mask_flatten.reshape(1, F_ * H * W)
        temporal_ctx = dict(
            frame_src=frame_src,
            frame_padding_mask=frame_padding_mask,
            spatial_shape=(H, W),
            valid_ratios=valid_ratios_frames[None, :, :],   # (1, F, 2)
        )

        x, ref_st, inter_st = self.st_branch(
            tgt, query_pos, reference_points, spatial_ctx, temporal_ctx)
        y, ref_ts, inter_ts = self.ts_branch(
            tgt, query_pos, reference_points, spatial_ctx, temporal_ctx)

        coupled = self.dafc(x, y)                       # (F, nq, C)
        final_ref = (ref_st + ref_ts) / 2.0             # (F, nq, 2)
        intermediates = inter_st + inter_ts
        return coupled, final_ref, intermediates


# ---------------------------------------------------------------------------
# Full transformer: vanilla encoder + ADTD
# ---------------------------------------------------------------------------
class ADTITransformer(nn.Module):
    def __init__(self, d_model=256, nhead=4, num_encoder_layers=6,
                 dim_feedforward=1024, dropout=0.1, activation='relu',
                 n_points=4, window=256, n_s_layers=2, n_t_layers=3,
                 n_levels_max=30):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = VanillaTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, nhead)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = VanillaTransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm)

        self.adtd = ADTD(d_model, dim_feedforward, dropout, activation,
                         nhead, n_points, window, n_s_layers, n_t_layers,
                         n_levels_max)

        # initial reference points from the positional queries
        self.reference_points = nn.Linear(d_model, 2)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # re-apply the specialized initializations (the loop above
        # overwrites them); reference_points keeps its xavier-initialized
        # weight (as in Deformable DETR) so that the initial reference
        # points of the queries are spread over the image
        for branch in (self.adtd.st_branch, self.adtd.ts_branch):
            for layer in branch.layers:
                attn = layer.cross_attn.inner if hasattr(
                    layer.cross_attn, 'inner') else layer.cross_attn
                attn._reset_parameters()
                nn.init.constant_(layer.bbox_embed.layers[-1].weight.data, 0.)
                nn.init.constant_(layer.bbox_embed.layers[-1].bias.data, 0.)
                nn.init.constant_(
                    layer.bbox_embed.layers[-1].bias.data[2:], -2.0)

    def forward(self, src_flatten, pos_flatten, mask_flatten, tgt,
                query_pos, spatial_shapes):
        """
        :param src_flatten   (F, hw, C) projected backbone features
        :param pos_flatten   (F, hw, C) positional encodings
        :param mask_flatten  (F, hw) padding mask
        :param tgt           (F, nq, C) initial content queries
        :param query_pos     (F, nq, C) positional queries
        :param spatial_shapes (h, w)
        :returns (coupled, final_ref, intermediates, memory)
        """
        memory = self.encoder(src_flatten, pos_flatten, mask_flatten)
        reference_points = self.reference_points(query_pos).sigmoid()
        coupled, final_ref, intermediates = self.adtd(
            tgt, query_pos, reference_points, memory, mask_flatten,
            spatial_shapes)
        return coupled, final_ref, intermediates, memory


def build_adti_transformer(args):
    return ADTITransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation='relu',
        n_points=args.dec_n_points,
        window=args.attn_window,
        n_s_layers=args.num_s_dtd_layers,
        n_t_layers=args.num_t_dtd_layers,
        n_levels_max=args.max_seq_frames,
    )
