import torch
import torchvision
from .mlgcn import MLGCN, MLGCNImproved, MLGCNEfficientNet, MLGCNRealTime, MLGCNDropout

model_dict = {'MLGCN':MLGCN,
              'MLGCNImproved':MLGCNImproved,
              'MLGCNEfficientNet': MLGCNEfficientNet,
              'MLGCNRealTime': MLGCNRealTime,
              'MLGCNDropout': MLGCNDropout
              }

def get_model(num_classes, args):
    if args.model_name not in model_dict:
        raise ValueError(f"Model '{args.model_name}' not found! Available models: {list(model_dict.keys())}")
    
    else:
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes, word_feature_path=args.word_feature_path)
        return model