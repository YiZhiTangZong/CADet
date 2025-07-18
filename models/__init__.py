from models.cadet import cadet

import torch

def create_model(
    args
) -> torch.nn.Module:
    """Models wrapper. Allows to create any model just with parametes

    """
    archs = [cadet]
    archs_dict = {a.__name__.lower(): a for a in archs}
    arch = args['architecture']
    try:
        model_class = archs_dict[arch.lower()]
    except KeyError:
        raise KeyError("Wrong architecture type `{}`. Available options are: {}".format(
            arch, list(archs_dict.keys()),
        ))
    if arch.lower() == 'cadet':
        return model_class( backbone = args.backbone, 
                            use_gau = args.use_gau, 
                            use_fim  =args.use_fim, 
                            up = args.up, 

                            module = args.module,
                            sat_pos = args.sat_pos,
                            use_fusion = args.use_fusion,
                            dual_decoder = args.dual_decoder,
                            dual_module = args.dual_module,
                            
                            classes = args.classes,
                            steps = args.steps,
                            reduce_dim = args.reduce_dim)
    else:
        raise RuntimeError('No implementation: ', arch.lower())