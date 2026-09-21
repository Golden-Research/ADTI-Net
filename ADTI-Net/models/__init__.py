# ------------------------------------------------------------------------
# ADTI-Net: video object detection
# ------------------------------------------------------------------------


def build_model(args):
    if getattr(args, 'model_type', 'transvod') == 'adti':
        # ADTI-Net (base) - self-contained, also works without the
        # compiled CUDA ops of models/ops
        from .adti_net import build as build_adti
        return build_adti(args)
    if args.dataset_file == "vid_single":
        from .deformable_detr_single import build as build_single
        return build_single(args)
    from .deformable_detr_multi import build as build_multi
    return build_multi(args)
