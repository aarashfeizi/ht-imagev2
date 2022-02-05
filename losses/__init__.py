import pytorch_metric_learning.losses as pml_losses

from losses import linkprediction, proxynca, triplet

LOSSES = {
    'pnpp': proxynca.ProxyNCA_prob,
    'bce': linkprediction.NormalBCELoss,
    'hardbce': linkprediction.HardBCELoss,
    # 'trpl': triplet.TripletMargin_Loss,
    'trpl': pml_losses.TripletMarginLoss,
    'proxy_nca': pml_losses.ProxyNCALoss,  # num_classes, embedding_size, softmax_scale=1,
    'proxy_anchor': pml_losses.ProxyAnchorLoss,  # num_classes, embedding_size, margin = 0.1, alpha = 32
    'arcface': pml_losses.ArcFaceLoss,  # num_classes, embedding_size, margin=28.6, scale=64,
    'angular': pml_losses.AngularLoss,  # alpha=40
    'circle': pml_losses.CircleLoss,  # m=0.4, gamma=80,
}

IMPLEMENTED_LOSSES = ['pnpp', 'bce', 'hardbce'] # 'trpl'


def get_inputs(**kwargs):
    to_ret_kwargs = {}
    for k, v in kwargs.items():
        if v is not None:
            to_ret_kwargs[k] = v

    return to_ret_kwargs


def get_loss(args):
    loss_name = args.get('loss')
    input_kwargs = None
    if loss_name in IMPLEMENTED_LOSSES:
        input_kwargs = get_inputs(args=args)
    elif loss_name == 'trpl':
        input_kwargs = get_inputs(margin=args.get('LOSS_margin'))
    elif loss_name == 'proxy_nca':
        input_kwargs = get_inputs(num_classes=args.get('nb_classes'),
                                  embedding_size=args.get('emb_size'),
                                  softmax_scale=args.get('NCA_scale'))

    elif loss_name == 'proxy_anchor':
        input_kwargs = get_inputs(num_classes=args.get('nb_classes'),
                                  embedding_size=args.get('emb_size'),
                                  margin=args.get('LOSS_margin'),
                                  alpha=args.get('LOSS_alpha'))
    elif loss_name == 'arcface':
        input_kwargs = get_inputs(num_classes=args.get('nb_classes'),
                                  embedding_size=args.get('emb_size'),
                                  margin=args.get('LOSS_margin'),
                                  scale=args.get('ARCFACE_scale'))
    elif loss_name == 'angular':
        input_kwargs = get_inputs(alpha=args.get('LOSS_alpha'))

    elif loss_name == 'circle':
        input_kwargs = get_inputs(m=args.get('CIR_m'),
                                  gamma=args.get('CIR_gamma'))

    if input_kwargs is None:
        raise Exception('Loss no supported on losses/__init__.py')

    to_ret = LOSSES[loss_name](**input_kwargs)

    return to_ret
