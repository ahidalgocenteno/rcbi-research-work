""""

This script trains a multi-label image classification model with uncertainty estimation using MC Dropout.
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
parser.add_argument('--model-name', type=str, default='MLGCNDropout')
parser.add_argument('--save-dir', default='./checkpoint/VOC2007/', type=str, help='save path')

''' Val or Tese setting '''
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

''' display '''
parser.add_argument('-d', '--display', dest='display', action='store_true', help='display mAP')
parser.add_argument('-s','--summary_writer', action='store_true',  default=False, help="start tensorboard")

''' loading of pre-trained model '''
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


def main(args):
    # Set the device
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

        # save the weights of the trained model
        model_path = os.path.join(args.save_dir, "model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    if args.pretrained != '':
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
        print('load pretrained model from {}'.format(args.pretrained))

    # Enable MC Dropout for uncertainty estimation
    model.activate_mc_dropout()

    def test_with_uncertainty_per_label(model, val_loader, num_classes, num_samples = 10):
        model.eval()  # Set the model to evaluation mode
        all_variances = torch.zeros(num_classes, device=device)
        all_entropies = torch.zeros(num_classes, device=device)
        label_counts = torch.zeros(num_classes, device=device)

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Testing Uncertainty"):
                images, labels = batch["image"].to(device), batch["target"].to(device)
                outputs = []

                # Perform multiple stochastic forward passes
                for _ in range(num_samples):
                    outputs.append(torch.sigmoid(model(images)))  # Convert to probabilities

                outputs = torch.stack(outputs, dim=0)  # [num_samples, batch_size, num_classes]
                mean_outputs = outputs.mean(dim=0)  # [batch_size, num_classes]
                variance = mean_outputs.var(dim=1).squeeze(0)
                entropy = -torch.sum(mean_outputs * torch.log(mean_outputs), dim=1)                
                # Update per-label metrics
                for i in range(labels.size(0)):  # Iterate over the batch
                    label_indices = labels[i] > 0  # Active labels for this sample
                    all_variances[label_indices] += variance[i]
                    all_entropies[label_indices] += entropy[i]
                    label_counts[label_indices] += 1

        # Normalize by the counts
        all_variances /= label_counts
        all_entropies /= label_counts

        return all_variances.cpu().numpy(), all_entropies.cpu().numpy()


    # Compute per-label uncertainty
    variances, entropies = test_with_uncertainty_per_label(model, val_loader, num_classes)

    # Save per-label uncertainty metrics
    results = {
        "variances": variances,
        "entropies": entropies,
    }

    # check if the folder uncertainty_results exists
    if not os.path.exists("./uncertainty_results/"):
        os.makedirs("./uncertainty_results/")
    save_path = os.path.join("./uncertainty_results/", args.data + "_per_label_uncertainty_mcdropout.pkl")
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
