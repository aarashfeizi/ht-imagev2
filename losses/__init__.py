
from losses import linkprediction, proxynca, triplet

LOSSES = {
    'pnpp': proxynca.ProxyNCA_prob,
    'bce': linkprediction.NormalBCELoss,
    'trpl': triplet.TripletMargin_Loss,
}

def get_loss(args):
    return LOSSES[args.get('loss')](args)