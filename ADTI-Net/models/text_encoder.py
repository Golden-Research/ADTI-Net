# ------------------------------------------------------------------------
# ADTI-Net (base model): video object detection.
# Textual feature encoder of the TFIL module.
#
# Reference:
#   "ADTI-Net: Alternating Decoupled Transformer Imitation Network for
#    Video Object Detection" (IEEE TIP).
#
# The paper encodes the class labels with the CLIP text encoder and maps
# them to a `text_dim`(=128)-d space through a textual projection head.
# If the `transformers` package or the pretrained CLIP weights are not
# available, a learnable class-embedding table is used as a fallback so
# that training can still proceed.
# ------------------------------------------------------------------------

import torch
from torch import nn


class TextEncoder(nn.Module):
    """Encode class names into textual features.

    Args:
        class_names: list of class names (index i <-> category id i+1).
        embed_dim:   dimension of the textual embedding space (paper: 128).
        model_name:  HF model id of the CLIP text encoder.
    """

    def __init__(self, class_names, embed_dim=128,
                 model_name='openai/clip-vit-base-patch32',
                 prompt_template='a photo of a {}'):
        super().__init__()
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.embed_dim = embed_dim
        self.using_clip = False

        try:
            from transformers import AutoTokenizer, CLIPTextModel
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            clip = CLIPTextModel.from_pretrained(model_name)
            clip.eval()
            for p in clip.parameters():
                p.requires_grad = False
            prompts = [prompt_template.format(n.replace('_', ' '))
                       for n in self.class_names]
            inputs = tokenizer(prompts, padding=True, return_tensors='pt')
            with torch.no_grad():
                feats = clip(**inputs).pooler_output        # [K, C_clip]
            self.clip_dim = int(feats.shape[-1])
            # frozen CLIP text features (buffer => moved with .to(device))
            self.register_buffer('raw_text_feats', feats)
            # textual projection head (trainable)
            self.proj = nn.Linear(self.clip_dim, embed_dim)
            self.using_clip = True
        except Exception as e:  # pragma: no cover
            print('[ADTI-Net] CLIP text encoder unavailable ({}: {}).'.format(
                type(e).__name__, e))
            print('[ADTI-Net] Falling back to learnable class embeddings. '
                  'Install `transformers` (with network access) to enable '
                  'CLIP textual features.')
            self.fallback_embed = nn.Embedding(self.num_classes, embed_dim)

    def forward(self):
        """Return the textual features of all classes: [num_classes, embed_dim]."""
        if self.using_clip:
            return self.proj(self.raw_text_feats)
        return self.fallback_embed.weight
