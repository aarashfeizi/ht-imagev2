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
    'supcon': pml_losses.SupConLoss, # temperature=0.1
    'multisim': pml_losses.MultiSimilarityLoss, # alpha=2, beta=50, base=0.5
    'lifted': pml_losses.LiftedStructureLoss, # neg_margin=1, pos_margin=0,
    'softtriple': pml_losses.SoftTripleLoss # num_classes, embedding_size, centers_per_class=10, la=20, gamma=0.1, margin=0.01
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
    elif loss_name == 'bce-trpl':
        input_kwargs = get_inputs(args=args,
                                  margin=args.get('LOSS_margin'))
    elif loss_name == 'supcon':
        input_kwargs = get_inputs(temperature=args.get('LOSS_temp'))
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
    elif loss_name == 'multisim':
        input_kwargs = get_inputs(alpha=args.get('LOSS_alpha'),
                                  beta=args.get('MS_beta'),
                                  base=args.get('MS_base'))
    elif loss_name == 'lifted':
        input_kwargs = get_inputs(neg_margin=args.get('LIFT_negmargin'),
                                  pos_margin=args.get('LIFT_posmargin'))
    elif loss_name == 'softtriple':
        input_kwargs = get_inputs(num_classes=args.get('nb_classes'),
                    embedding_size=args.get('emb_size'),
                    centers_per_class=args.get('SOFTTRPL_cpc'),
                    la=args.get('SOFTTRPL_lambda'),
                    gamma=args.get('SOFTTRPL_gamma'),
                    margin=args.get('LOSS_margin'))

    if input_kwargs is None:
        raise Exception('Loss no supported on losses/__init__.py')

    to_ret = LOSSES[loss_name](**input_kwargs)

    # if args.get('with_bce'):
    #     assert loss_name != 'bce'
    #     assert loss_name != 'hardbce'
    #
    #     to_ret = linkprediction.BceAndOtherLoss(args=args, other_loss=to_ret)

    return to_ret
