import torch
import torchvision
from .mlgcn import MLGCN, MLGCNImproved

model_dict = {'MLGCN':MLGCN,
              'MLGCNImproved':MLGCNImproved}

def get_model(num_classes, args):
    if args.model_name not in model_dict:
        raise ValueError(f"Model '{args.model_name}' not found! Available models: {list(model_dict.keys())}")
    
    if args.model_name == 'MLGCN':
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](res101, num_classes, word_feature_path=args.word_feature_path)
        return model
    elif args.model_name == 'MLGCNImproved':
        res101 = torchvision.models.resnet101(pretrained=True)
        model = model_dict[args.model_name](num_classes, word_feature_path=args.word_feature_path)
        return model