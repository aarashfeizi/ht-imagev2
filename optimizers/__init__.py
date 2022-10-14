import torch

from optimizers import sam

OPTIMIZERS = {'adam': torch.optim.Adam,
              'sam': sam.SAM,
              'sgd': torch.optim.SGD}


def get_optimizer(args, optimizer_name, learnable_params):
    opt = None
    if optimizer_name != 'sam':
        opt = OPTIMIZERS[optimizer_name](params=learnable_params,
                                         lr=args.get('learning_rate'),
                                         weight_decay=args.get('weight_decay'))
    elif optimizer_name == 'adam': # optimizer is sam
        base_optimizer = OPTIMIZERS['sgd']
        opt = OPTIMIZERS[optimizer_name](params=learnable_params,
                                         base_optimizer=base_optimizer,
                                         lr=args.get('learning_rate'),
                                         weight_decay=args.get('weight_decay'),
                                         momentum=0.9)
    elif optimizer_name == 'sgd':
        opt = OPTIMIZERS[optimizer_name](params=learnable_params,
                                        lr=args.get('learning_rate'),
                                        weight_decay=args.get('weight_decay'),
                                        momentum=args.get('opt_momentum'))

    if opt is None:
        raise Exception(f'Optimizer {optimizer_name} is not defnied')

    return opt