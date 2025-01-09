import torch.optim as optim 


def get_optimizer_grouped_params_with_llrd(cfg, model):
    ''' layerwise learning rate decay '''
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']

    # initialize lr for task specific layer 
    optimizer_grouped_parameters = [
        {
            'params':[p for n, p in model.named_parameters() if 'backbone' not in n],
            'lr': cfg.optimizer.head_lr,
            'weight_decay': cfg.optimizer.weight_decay
        }
    ]

    layers = [model.backbone.embeddings] + list(model.backbone.encoder.layer)
    layers.reverse()
    lr = cfg.optimizer.lr

    for layer in layers:
        lr *= cfg.optimizer.llrd

        optimizer_grouped_parameters += [
            {
                'params': [p for n, p in layer.named_parameters() if not (nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': cfg.optimizer.weight_decay

            }, {
                'params': [p for n, p in layer.named_parameters() if (nd in n for nd in no_decay)],
                'lr': lr,
                'weight_decay': 0.0
            }
        ]

    return optimizer_grouped_parameters


def get_optimizer_grouped_params_with_no_llrd(cfg, model):
    ''' layerwise learning rate decay '''
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    backbone_params = model.backbone.named_parameters()


    optimizer_grouped_parameters = [
        {
            'params':[p for n, p in model.named_parameters() if 'backbone' not in n],
            'lr': cfg.optimizer.head_lr,
            'weight_decay': cfg.optimizer.weight_decay
        },
        {
            'params': [p for n, p in backbone_params if not (nd in n for nd in no_decay)],
            'lr': cfg.optimizer.lr,
            'weight_decay': cfg.optimizer.weight_decay

        }, {
            'params': [p for n, p in backbone_params if (nd in n for nd in no_decay)],
            'lr': cfg.optimizer.lr,
            'weight_decay': 0.0
        }
    ]

    return optimizer_grouped_parameters



def get_optimizer(cfg, model):
    
    '''
    Step 1 - retrieve the parameters of model which has its own learning rate and weight decay 
    Step 2 - retrieve the parameter of the optimizer (beta1, beta2, eps, lr, weight_decay)
    '''

    if cfg.optimizer.use_llrd:
        optimizer_grouped_parameters = get_optimizer_grouped_params_with_llrd(cfg, model)
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_params_with_no_llrd(cfg, model)


    optimizer_params = {
        'betas': (cfg.optimizer.beta1,cfg.optimizer.beta2),
        'eps': cfg.optimizer.eps,
        'lr': cfg.optimizer.lr
    }


    if cfg.optimizer.use_bnb:
        import bitsandbytes as bnb 
        return bnb.optim.AdamW8bit(
            optimizer_grouped_parameters,
            **optimizer_params
        )
    else:
        return optim.AdamW(
            optimizer_grouped_parameters,
            **optimizer_params
        )
