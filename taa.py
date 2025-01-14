""""

This script trains a multi-label image classification model with uncertainty estimation using Temporal Average Augmentation (TTA).
The model is trained on the given dataset and the per-label uncertainty metrics are computed on the validation set.
The per-label uncertainty metrics are saved to a file named "per_label_uncertainty.pkl".

"""


import os
import torch
import argparse
from tqdm import tqdm
from models import get_model 
from data import make_data_loader
from trainer import Trainer
import pickle
import random
import warnings
import torch.backends.cudnn as cudnn

import albumentations as A
from torchvision.transforms.functional import to_tensor

# Argument parser setup
parser = argparse.ArgumentParser(description='PyTorch Multi-label Image Classification with Uncertainty')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./dataset/', type=str, help='save path')
parser.add_argument('--image-size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--epoch_step', default=[30, 40], type=int, nargs='+', help='number of epochs to change learning rate')  
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=200, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--num_classes', default=20, type=int, help='the number of the classses')
parser.add_argument('--batch-size', default=16, type=int, help='The batch size used')
parser.add_argument('-o', '--optimizer', default='SGD', type=str, help="The optimizer can be only chosen from {\'SGD\', \'Adam\', \'AdamW\'} for now. More may be implemented later")
parser.add_argument('-backbone','--backbone', default='ResNet101', type=str, help='ResNet101, resnet101, ResNeXt50-swsl, ResNeXt50_32x4d (default: ResNet101)')
# parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--warmup_epoch',  default=2, type=int, help='WarmUp epoch')
parser.add_argument('-up','--warmup_scheduler', action='store_true', default=False, help='star WarmUp')
parser.add_argument('--word_feature_path', default='./wordfeature/', type=str, help='word feature path')



''' Train setting '''
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014, VOC2007, VOC2012, VG_100K, CoCoDataset, nuswide, mirflickr25k')
parser.add_argument('--model-name', type=str, default='MLGCN')
parser.add_argument('--save-dir', default='./checkpoint/VOC2007/', type=str, help='save path')

''' Val or Tese setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

''' display '''
parser.add_argument('-d', '--display', dest='display', action='store_true', help='display mAP')
parser.add_argument('-s','--summary_writer', action='store_true',  default=False, help="start tensorboard")

''' loading of pre-trained model '''
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


def compute_tta_uncertainty(model, loader, num_samples, num_classes, device, single_image=None):
    """
    Perform Test-Time Augmentation (TTA) and compute per-label uncertainty.
    
    Args:
        model: PyTorch model for inference.
        loader: DataLoader for validation data. If processing a single image, pass None.
        num_samples: Number of TTA samples per image.
        num_classes: Number of output classes.
        device: Device for computation.
        single_image: (Optional) Single image tensor to process. If None, processes the entire loader.
    
    Returns:
        results: Dictionary containing label-wise variances and entropies.
    """
    # Define TTA augmentations
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.5),
    ])

    model.eval()
    all_variances = torch.zeros(num_classes, device=device)
    all_entropies = torch.zeros(num_classes, device=device)
    label_counts = torch.zeros(num_classes, device=device)

    def process_image(image):
        """Helper function to apply TTA and compute uncertainty for a single image."""
        predictions = []
        for _ in range(num_samples):
            if isinstance(image, torch.Tensor):
                image = image.permute(1, 2, 0).cpu().numpy()  # Convert to HWC format for Albumentations
            
            augmented_image = tta_transforms(image=image)["image"]  # Apply TTA
            augmented_image = to_tensor(augmented_image)  # Convert to tensor
            
            if augmented_image.ndimension() == 3 and augmented_image.shape[0] != 3:  # Ensure CHW format
                augmented_image = augmented_image.permute(2, 0, 1)
            
            augmented_image = augmented_image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            
            predictions.append(torch.sigmoid(model(augmented_image)))

        predictions = torch.stack(predictions, dim=0)  # [num_samples, 1, num_classes]
        mean_prediction = predictions.mean(dim=0).squeeze(0)
        entropy = -torch.sum(mean_prediction * torch.log(mean_prediction))  # Scalar entropy
        variance = mean_prediction.var()
        return variance.cpu(), entropy.cpu(), mean_prediction.cpu()


    with torch.no_grad():
        if single_image is not None:
            # Process a single image
            variance, entropy = process_image(single_image)
            return {"variances": variance.cpu().tolist(), "entropies": entropy.cpu().tolist()}
        
        # Process the entire loader
        for batch in tqdm(loader, desc="TTA Testing"):
            images, labels = batch["image"].to(device), batch["target"].to(device)

            for i in range(images.size(0)):  # Process each image in the batch
                image = images[i]
                label = labels[i]
                variance, entropy, mean_prediction = process_image(image)

                # check if the prediction is correct
                if torch.all(torch.eq(mean_prediction.argmax(), label.argmax())):
                    # Update per-label metrics
                    label_indices = label > 0  # Active labels
                    all_variances[label_indices] += variance
                    all_entropies[label_indices] += entropy
                    label_counts[label_indices] += 1

    # Normalize by counts (obtain the mean od the variance and entropy for each label)
    all_variances /= label_counts
    all_entropies /= label_counts

    return all_variances.cpu().tolist(), all_entropies.cpu().tolist()


def main(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print("Using device:", torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU")

    if args.seed is not None:
        print ('* absolute seed: {}'.format(args.seed))
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    is_train = True if not args.evaluate else False
    train_loader, val_loader, num_classes = make_data_loader(args, is_train=is_train)
    if is_train == True:
        args.iter_per_epoch = len(train_loader)
    else:
        args.iter_per_epoch = 1000 # randm

    # Initialize the model
    model = get_model(num_classes, args)
    model.to(device)

    # Initialize the trainer
    criterion = torch.nn.MultiLabelSoftMarginLoss()
    trainer = Trainer(model, criterion, train_loader, val_loader, args)

    # Train or evaluate
    if not args.evaluate and args.pretrained == '':
        trainer.train(device)

    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print('load pretrained model from {}'.format(args.pretrained))

    # Compute per-label uncertainty
    variances, entropies = compute_tta_uncertainty(model, val_loader, num_samples=10, num_classes=num_classes, device=device)

    # Save per-label uncertainty metrics
    results = {
        "variances": variances,
        "entropies": entropies
    }
    # check if the folder uncertainty_results exists
    if not os.path.exists("./uncertainty_results/"):
        os.makedirs("./uncertainty_results/")
    save_path = os.path.join("./uncertainty_results/", args.data + "_per_label_uncertainty_taa.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(results, f)

    print("Per-label variances:", variances)
    print("Per-label entropies:", entropies)

if __name__ == "__main__":
    args = parser.parse_args()
    args.data_root_dir='dataset'
    backbone = {1:'ResNet101'}
    args.backbone = backbone[1]
    torch.cuda.set_device(args.gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    args.seed = 1 # seed
    args.epochs = 50 #
    args.optimizer = {1:'SGD', 2: 'Adam', 3:'AdamW'}[1]
    args.display_interval = 1000
    args.warmup_scheduler = {1: False, 2: True}[2]
    args.warmup_epoch = 0 if args.warmup_scheduler == False else args.warmup_epoch
    args.word_feature_path = os.path.join(os.getcwd(), 'wordfeature')

    if args.optimizer == 'SGD':
        args.lr = 0.01 # voc is 0.01 and coco is 0.05
        args.lrp = 0.1
        args.epoch_step = [25, 35] # cutout

    elif args.optimizer == 'Adam':
        args.lr = 5 * 1e-5
        args.lrp = 0.1
        args.weight_decay = .0

    elif args.optimizer == 'AdamW':
        args.lr = 5 * 1e-5
        args.lrp = 0.01
        args.weight_decay = 1e-4
        args.epoch_step = [10, 20]

    if args.evaluate == True:
        args.image_size = 576
    else:
        args.image_size = 448
        args.resume=''

    main(args)
