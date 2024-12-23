import argparse
import os
from data import make_data_loader
import torch
import numpy as np
import pandas as pd

# VOC2017 class labels
VOC_LABELS = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

parser = argparse.ArgumentParser(description='PyTorch Training for Multi-label Image Classification')

''' Fixed in general '''
parser.add_argument('--data_root_dir', default='./dataset/', type=str, help='save path')
parser.add_argument('--image-size', '-i', default=448, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--epoch_step', default=[30, 40], type=int, nargs='+', help='number of epochs to change learning rate')  
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('-j', '--num_workers', default=4, type=int, metavar='INT', help='number of data loading workers (default: 4)')
parser.add_argument('--display_interval', default=200, type=int, metavar='M', help='display_interval')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float, metavar='LRP', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max_clip_grad_norm', default=10.0, type=float, metavar='M', help='max_clip_grad_norm')
parser.add_argument('--seed', default=1, type=int, help='seed for initializing training. ')
parser.add_argument('--num_classes', default=20, type=int, help='the number of the classses')
parser.add_argument('-o', '--optimizer', default='SGD', type=str, help="The optimizer can be only chosen from {\'SGD\', \'Adam\', \'AdamW\'} for now. More may be implemented later")
parser.add_argument('-backbone','--backbone', default='ResNet101', type=str, help='ResNet101, resnet101, ResNeXt50-swsl, ResNeXt50_32x4d (default: ResNet101)')
# parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--warmup_epoch',  default=2, type=int, help='WarmUp epoch')
parser.add_argument('-up','--warmup_scheduler', action='store_true', default=False, help='star WarmUp')
parser.add_argument('--word_feature_path', default='./wordfeature/', type=str, help='word feature path')
parser.add_argument('--data', metavar='NAME', help='dataset name (e.g. COCO2014, VOC2007, VOC2012, VG_100K, CoCoDataset, nuswide, mirflickr25k')
parser.add_argument('--model_name', type=str, default='GNN')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')


def main(args):

    train_loader, val_loader, num_classes = make_data_loader(args, is_train=False)

    num_classes = 20

    tau_threshold = 0.3
    p = 0.2

    data = next(iter(val_loader))

    cooccurrence_matrix = np.zeros((num_classes, num_classes), dtype=int)
    label_frequencies = np.zeros(num_classes, dtype=int)

    for data in val_loader:

        image = data['image']
        labels = data['target']

        # Find indices where value is 1
        indices = torch.nonzero(labels == 1, as_tuple=True)[1]

        # Convert indices to a list
        indices_list = indices.tolist()

        # Update individual label frequencies
        for label in indices_list:
            label_frequencies[label] += 1

         # For each label in the current sample
        for i in indices_list:
            for j in indices_list:
                # if i != j:
                cooccurrence_matrix[i, j] += 1

    prob_matrix = np.zeros_like(cooccurrence_matrix, dtype=float)
    # Calculate conditional probabilities
    for i in range(num_classes):
        if label_frequencies[i] > 0:  # Avoid division by zero
            # Al hacerlo asi los valores importantes lo tienen las columns
            prob_matrix[i, :] = cooccurrence_matrix[i, :] / label_frequencies[i]

    # cooccurrence_df = pd.DataFrame(cooccurrence_matrix)
    prob_df = pd.DataFrame(prob_matrix, index=VOC_LABELS, columns=VOC_LABELS)

    # Correlation matrix binarization
    A = (prob_matrix > tau_threshold).astype('float32')
    A_df = pd.DataFrame(A, index=VOC_LABELS, columns=VOC_LABELS)

    A_prime = np.zeros_like(A, dtype=float)

    # Calculate off-diagonal elements
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                # Sum of all connections from i to other nodes (excluding j)
                sum_connections = sum(A[i, k] for k in range(num_classes) if k != i)
                
                # Apply the formula for off-diagonal elements
                if sum_connections > 0:  # Avoid division by zero
                    A_prime[i, j] = p * A[i, j] / sum_connections
                else:
                    A_prime[i, j] = 0
            else:
                # Diagonal elements are 1-p
                A_prime[i, j] = 1 - p


    A_prime_df = pd.DataFrame(A_prime, index=VOC_LABELS, columns=VOC_LABELS)
    print("hola")



if __name__ == "__main__":
    args = parser.parse_args()
    args.data_root_dir='dataset'
    dataset_name = {1:'COCO2014', 2:'VOC2007', 3:'VOC2012'}
    backbone = {1:'ResNet101'}
    args.data = dataset_name[2]
    args.backbone = backbone[1]
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

    work = 'SGD_COCO_lr_001_lrp_01_bs16_5'
    args.save_dir = './checkpoint/' + args.data + '/' +args.model_name+'/' + work

    if args.evaluate == True:
        args.image_size = 576
    else:
        args.image_size = 448
        args.resume=''

    args.batch_size = 1


    main(args)
