import os
import torch
import torch.nn as nn
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import timm

import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

def gen_A(num_classes, t, adj_file):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int32)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MLGCN(nn.Module):
    def __init__(self, model, num_classes, word_feature_path):
        super(MLGCN, self).__init__()
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.num_classes = num_classes
        self.word_feature_path = word_feature_path
        self.pooling = nn.AdaptiveMaxPool2d((1,1))

        # word2vector
        if self.num_classes == 80:
            self.word_features = os.path.join(self.word_feature_path, 'coco_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'coco_adj.pkl')
        elif self.num_classes == 81:
            self.word_features = os.path.join(self.word_feature_path, 'nuswide_glove_word2vec.npy')
            self.adj_file = os.path.join(self.word_feature_path, 'nus_adj.pkl')
            self.t = 0.4
        else:
            self.word_features = os.path.join(self.word_feature_path, 'voc_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'voc_adj.pkl')
            self.t = 0.4

        self.word_feature_dim = 300 # 2048
        self.image_feature_dim = 1024 # 1024

        # self.word_features = self.load_features()
        
        with open(self.word_features, 'rb') as point:
            print('graph input: loaded from {}'.format(self.word_features))
            import pickle
            word_features = pickle.load(point)
        self.word_features = torch.from_numpy(word_features).float()
            

        # input_dim, output_dim, support_num, dropout=0
        self.gc1 = GraphConvolution(self.word_feature_dim, 1024)
        self.gc2 = GraphConvolution(1024, 2048)

        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t=0.4, adj_file=self.adj_file)
        self.A = nn.Parameter(torch.from_numpy(_adj).float())
        self.adj = gen_adj(self.A)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def load_features(self):
        return nn.Parameter(torch.from_numpy(np.load(self.word_features)).float(), requires_grad=False)
        # return  torch.from_numpy(np.load(self.word_features)).float()

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # inp = self.word_features.detach()
        # print(self.word_features.requires_grad)
        # inp = self.word_features.clone().cuda() #.detach()
        inp = self.word_features.cuda().detach()
        adj = self.adj.cuda().detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    # def get_config_optim(self, lr, lrp):
    #     small_lr_layers = [p for n, p in self.features.named_parameters() if "features" in n and p.requires_grad]
    #     large_lr_layers = [p for n, p in self.named_parameters() if "features" not in n and p.requires_grad]        
    #     return [
    #             {'params': small_lr_layers, 'lr': lr * lrp},
    #             {'params': large_lr_layers, 'lr': lr},
    #             ]
    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]
    
class MLGCNImproved(nn.Module):
    def __init__(self, dummy_model, num_classes, word_feature_path):
        super(MLGCNImproved, self).__init__()

        model = timm.create_model('resnext50_32x4d', pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])

        self.num_classes = num_classes
        self.word_feature_path = word_feature_path
        self.pooling = nn.AdaptiveMaxPool2d((1,1))

        # word2vector
        if self.num_classes == 80:
            self.word_features = os.path.join(self.word_feature_path, 'coco_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'coco_adj.pkl')
        elif self.num_classes == 81:
            self.word_features = os.path.join(self.word_feature_path, 'nuswide_glove_word2vec.npy')
            self.adj_file = os.path.join(self.word_feature_path, 'nus_adj.pkl')
            self.t = 0.4
        else:
            self.word_features = os.path.join(self.word_feature_path, 'voc_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'voc_adj.pkl')
            self.t = 0.4

        self.word_feature_dim = 300 # 2048
        self.image_feature_dim = 1024 # 1024

        # self.word_features = self.load_features()
        
        with open(self.word_features, 'rb') as point:
            print('graph input: loaded from {}'.format(self.word_features))
            import pickle
            word_features = pickle.load(point)
        self.word_features = torch.from_numpy(word_features).float()
            

        # input_dim, output_dim, support_num, dropout=0
        self.gc1 = GraphConvolution(self.word_feature_dim, 1024)
        self.gc2 = GraphConvolution(1024, 2048)

        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t=0.4, adj_file=self.adj_file)
        self.A = nn.Parameter(torch.from_numpy(_adj).float())
        self.adj = gen_adj(self.A)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def load_features(self):
        return nn.Parameter(torch.from_numpy(np.load(self.word_features)).float(), requires_grad=False)
        # return  torch.from_numpy(np.load(self.word_features)).float()

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # inp = self.word_features.detach()
        # print(self.word_features.requires_grad)
        # inp = self.word_features.clone().cuda() #.detach()
        inp = self.word_features.cuda().detach()
        adj = self.adj.cuda().detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    # def get_config_optim(self, lr, lrp):
    #     small_lr_layers = [p for n, p in self.features.named_parameters() if "features" in n and p.requires_grad]
    #     large_lr_layers = [p for n, p in self.named_parameters() if "features" not in n and p.requires_grad]        
    #     return [
    #             {'params': small_lr_layers, 'lr': lr * lrp},
    #             {'params': large_lr_layers, 'lr': lr},
    #             ]
    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]

class MLGCNEfficientNet(nn.Module):
    def __init__(self, dummy_model ,num_classes, word_feature_path):
        super(MLGCNEfficientNet, self).__init__()

        model = timm.create_model('tf_efficientnetv2_xl', pretrained=True)

        # Extract all layers except the last two (classifier and global pooling)
        self.features = nn.Sequential(*list(model.children())[:-2])

        self.num_classes = num_classes
        self.word_feature_path = word_feature_path
        self.pooling = nn.AdaptiveMaxPool2d((1,1))

        # word2vector
        if self.num_classes == 80:
            self.word_features = os.path.join(self.word_feature_path, 'coco_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'coco_adj.pkl')
        elif self.num_classes == 81:
            self.word_features = os.path.join(self.word_feature_path, 'nuswide_glove_word2vec.npy')
            self.adj_file = os.path.join(self.word_feature_path, 'nus_adj.pkl')
            self.t = 0.4
        else:
            self.word_features = os.path.join(self.word_feature_path, 'voc_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'voc_adj.pkl')
            self.t = 0.4

        self.word_feature_dim = 300 # 2048
        self.image_feature_dim = 1024 # 1024

        # self.word_features = self.load_features()
        
        with open(self.word_features, 'rb') as point:
            print('graph input: loaded from {}'.format(self.word_features))
            import pickle
            word_features = pickle.load(point)
        self.word_features = torch.from_numpy(word_features).float()
            

        # input_dim, output_dim, support_num, dropout=0
        self.gc1 = GraphConvolution(self.word_feature_dim, 1024)
        self.gc2 = GraphConvolution(1024, 1280)

        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t=0.4, adj_file=self.adj_file)
        self.A = nn.Parameter(torch.from_numpy(_adj).float())
        self.adj = gen_adj(self.A)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def load_features(self):
        return nn.Parameter(torch.from_numpy(np.load(self.word_features)).float(), requires_grad=False)
        # return  torch.from_numpy(np.load(self.word_features)).float()

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # inp = self.word_features.detach()
        # print(self.word_features.requires_grad)
        # inp = self.word_features.clone().cuda() #.detach()
        inp = self.word_features.cuda().detach()
        adj = self.adj.cuda().detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    # def get_config_optim(self, lr, lrp):
    #     small_lr_layers = [p for n, p in self.features.named_parameters() if "features" in n and p.requires_grad]
    #     large_lr_layers = [p for n, p in self.named_parameters() if "features" not in n and p.requires_grad]        
    #     return [
    #             {'params': small_lr_layers, 'lr': lr * lrp},
    #             {'params': large_lr_layers, 'lr': lr},
    #             ]
    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]
    
class MLGCNRealTime(nn.Module):
    def __init__(self, dummy_model, num_classes, word_feature_path):
        super(MLGCNRealTime, self).__init__()

        model = timm.create_model('mobilenetv3_large_100', pretrained=True)
        self.features = nn.Sequential(*list(model.children())[:-2])

        self.num_classes = num_classes
        self.word_feature_path = word_feature_path
        self.pooling = nn.AdaptiveMaxPool2d((1,1))

        # word2vector
        if self.num_classes == 80:
            self.word_features = os.path.join(self.word_feature_path, 'coco_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'coco_adj.pkl')
        elif self.num_classes == 81:
            self.word_features = os.path.join(self.word_feature_path, 'nuswide_glove_word2vec.npy')
            self.adj_file = os.path.join(self.word_feature_path, 'nus_adj.pkl')
            self.t = 0.4
        else:
            self.word_features = os.path.join(self.word_feature_path, 'voc_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'voc_adj.pkl')
            self.t = 0.4

        self.word_feature_dim = 300 # 2048
        self.image_feature_dim = 1024 # 1024

        # self.word_features = self.load_features()
        
        with open(self.word_features, 'rb') as point:
            print('graph input: loaded from {}'.format(self.word_features))
            import pickle
            word_features = pickle.load(point)
        self.word_features = torch.from_numpy(word_features).float()
            

        # input_dim, output_dim, support_num, dropout=0
        self.gc1 = GraphConvolution(self.word_feature_dim, 1024)
        self.gc2 = GraphConvolution(1024, 1280)

        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t=0.4, adj_file=self.adj_file)
        self.A = nn.Parameter(torch.from_numpy(_adj).float())
        self.adj = gen_adj(self.A)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def load_features(self):
        return nn.Parameter(torch.from_numpy(np.load(self.word_features)).float(), requires_grad=False)
        # return  torch.from_numpy(np.load(self.word_features)).float()

    def forward_feature(self, x):
        x = self.features(x)
        return x

    def forward(self, x):
        feature = self.forward_feature(x)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)

        # inp = self.word_features.detach()
        # print(self.word_features.requires_grad)
        # inp = self.word_features.clone().cuda() #.detach()
        inp = self.word_features.cuda().detach()
        adj = self.adj.cuda().detach()

        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    # def get_config_optim(self, lr, lrp):
    #     small_lr_layers = [p for n, p in self.features.named_parameters() if "features" in n and p.requires_grad]
    #     large_lr_layers = [p for n, p in self.named_parameters() if "features" not in n and p.requires_grad]        
    #     return [
    #             {'params': small_lr_layers, 'lr': lr * lrp},
    #             {'params': large_lr_layers, 'lr': lr},
    #             ]
    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p:id(p) not in small_lr_layers, self.parameters())
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': large_lr_layers, 'lr': lr},
                ]


class MLGCNDropout(nn.Module):
    def __init__(self, model, num_classes, word_feature_path, dropout_p=0.5, test_efficient=False):
        super(MLGCNDropout, self).__init__()

        # Choose feature extractor
        if test_efficient:
            model = timm.create_model('mobilenetv3_large_100', pretrained=True)
            self.features = nn.Sequential(*list(model.children())[:-2])
            self.image_feature_dim = 1280  # Mobilenetv3 output dimension
        else:
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
            )
            self.image_feature_dim = 2048  # ResNet output dimension

        self.num_classes = num_classes
        self.word_feature_path = word_feature_path
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        # word2vector
        if self.num_classes == 80:
            self.word_features = os.path.join(self.word_feature_path, 'coco_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'coco_adj.pkl')
        elif self.num_classes == 81:
            self.word_features = os.path.join(self.word_feature_path, 'nuswide_glove_word2vec.npy')
            self.adj_file = os.path.join(self.word_feature_path, 'nus_adj.pkl')
            self.t = 0.4
        else:
            self.word_features = os.path.join(self.word_feature_path, 'voc_glove_word2vec.pkl')
            self.adj_file = os.path.join(self.word_feature_path, 'voc_adj.pkl')
            self.t = 0.4

        self.word_feature_dim = 300
        self.gcn_hidden_dim = 1024
        self.gcn_output_dim = 2048

        # Adjust feature extractor output to match GCN 
        if self.image_feature_dim != self.gcn_output_dim:
            self.feature_transform = nn.Linear(self.image_feature_dim, self.gcn_output_dim)

        # Load word features
        with open(self.word_features, 'rb') as point:
            print(f'Graph input: loaded from {self.word_features}')
            import pickle
            word_features = pickle.load(point)
        self.word_features = torch.from_numpy(word_features).float()

        # Graph Convolutional layers
        self.gc1 = GraphConvolution(self.word_feature_dim, self.gcn_hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.gc2 = GraphConvolution(self.gcn_hidden_dim, self.gcn_output_dim)
        self.dropout2 = nn.Dropout(p=dropout_p)

        # Activation
        self.relu = nn.LeakyReLU(0.2)

        # Adjacency matrix
        _adj = gen_A(num_classes, t=0.4, adj_file=self.adj_file)
        self.A = nn.Parameter(torch.from_numpy(_adj).float())
        self.adj = gen_adj(self.A)

    def forward_feature(self, x):
        x = self.features(x)  # Feature extractor
        x = self.pooling(x)  # Adaptive pooling
        x = x.view(x.size(0), -1)  # Flatten
        if hasattr(self, 'feature_transform'):
            x = self.feature_transform(x)  # Transform feature dimension
        return x

    def forward(self, x):
        # Extract image features
        feature = self.forward_feature(x)

        # Prepare graph input
        inp = self.word_features.cuda().detach()
        adj = self.adj.cuda().detach()

        # Pass through GCN layers
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.gc2(x, adj)
        x = self.dropout2(x)

        # Combine image and GCN features
        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        return x

    def activate_mc_dropout(self):
        """Activate dropout layers during test time for MC Dropout."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def get_config_optim(self, lr, lrp):
        small_lr_layers = list(map(id, self.features.parameters()))
        large_lr_layers = filter(lambda p: id(p) not in small_lr_layers, self.parameters())
        return [
            {'params': self.features.parameters(), 'lr': lr * lrp},
            {'params': large_lr_layers, 'lr': lr},
        ]

def test():
    import os
    import torch
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    from torchvision import models, transforms
    import torchvision
    lr = 0.05
    lrp = 0.1

    resnet101 = torchvision.models.resnet101(pretrained=True)
    Net = MLGCN(resnet101, num_classes=80).cuda()

    # count = 0
    # for name, param in Net.named_parameters():
    #     print(count,': ', name)
    #     count += 1
    # print('=='*30)

    # param_dicts = [
    #     {"params": Net.get_config_optim(lr, lrp)},
    # ]

    # Net.get_config_optim(lr, lrp)


    # input image
    img = torch.randn(1, 3, 448, 448).cuda()

    out = Net(img)
    print(out.shape)




if __name__ == '__main__':

    test()
