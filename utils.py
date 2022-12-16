from PIL import Image
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageOps
import os
import shutil
import pandas as pd
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data import Subset, DataLoader, ConcatDataset
from torchvision import datasets, transforms
import torch
import torch.backends.cudnn as cudnn
import seaborn as sns
import matplotlib.legend
import scipy.ndimage.interpolation as interpolation
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc,accuracy_score, precision_score
import csv

def create_rotates(r):
    """Creates vector of rotation degrees compatible with scipy' rotate.
    Args:
    r: param r for rotation augmentation attack. Defines max rotation by +/-r. Leads to 2*r+1 total images per sample.
    Returns: vector of rotation degrees compatible with scipy' rotate.
    """
    if r is None:
        return None
    if r == 1:
        return [0.0]
    # rotates = [360. / r * i for i in range(r)]
    rotates = np.linspace(-r, r, (r * 2 + 1))
    print('Number of augmentations : '+ str(len(rotates)))
    return rotates

def apply_augment(ds, augment, type_):
    """Applies an augmentation from create_rotates or create_translates.
        Args:
        ds: tuple of (images, labels) describing a datset. Images should be 4D of (N,H,W,C) where N is total images.
        augment: the augment to apply. (one element from augments returned by create_rotates/translates)
        type_: attack type, either 'd' or 'r'
        Returns:
    """
    print(ds[0].shape)
    if type_ == 'd':
        ds = (interpolation.shift(ds[0], augment, mode='nearest'), ds[1])
    else:
        ds = (interpolation.rotate(ds[0], augment, (1, 2), reshape=False), ds[1])
    return ds

def load_dataset(args, dataset, cluster=None, mode = 'target', max_num = 100):
    kwargs = {'num_workers': 2, 'pin_memory': True}
    
    transform=[]
    if dataset == 'GTSRB':
        transform = [transforms.Resize((64,64))]
    elif dataset == 'Face':

        transform = [transforms.Resize((128,128))]
   
    if args.augmentation and mode!= 'adversary':
        
        print("%"*10+"  Augmeting the data  "+"%"*10)
        transform=transform+[Rand_Augment()]

    transform=transforms.Compose(transform+[transforms.ToTensor()])
            
    if dataset == 'CIFAR10':
        print("Loding CIFAR10 Dataset")
        whole_set = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        max_cluster = 3000
        test_size = 1000
    elif dataset == 'CIFAR100':
        whole_set = datasets.CIFAR100('data', train=True, download=True, transform=transform)
        max_cluster = 7000
        test_size = 1000
        print('*'*15+'   The size of whole datasat : '+str(len(whole_set))+'  '+'*'*10)
    elif dataset == 'GTSRB':
        whole_set = datasets.ImageFolder('data/GTSRB/', transform= transform)
        max_cluster = 600
        test_size = 500
        print('*'*15+'   The size of whole datasat : '+str(len(whole_set))+'  '+'*'*10)
    elif dataset == 'Face':
        whole_set = datasets.ImageFolder('data/lfw/', transform=transform)
        max_cluster = 350
        test_size = 100
        print('*'*15+'   The size of whole datasat : '+str(len(whole_set))+'  '+'*'*10)
    
    elif dataset == 'purchase':
        x, y = [], []
        with open('data/purchase', 'r') as infile:
            reader = csv.reader(infile)
            for line in reader:
                y.append(int(line[0]))
                x.append([int(x) for x in line[1:]])
        x = np.array(x).astype(np.float32)
        y = (np.array(y) - 1).reshape((-1, 1)).squeeze()
        
        cluster=10000
        test_size=10000
        np.random.seed(456)
        indices = np.arange(len(y))
        p = np.random.permutation(indices)
        x,y=x[indices],y[indices]
        whole_set=[(x[i],y[i]) for i in range(len(y))]
        
    elif dataset == 'texas':
        x, y = [], []
        with open('data/texas/100/feats', 'r') as infile:
            reader = csv.reader(infile)
            for line in reader:
                x.append([int(x) for x in line[1:]])
            x = np.array(x).astype(np.float32)
        with open('data/texas/100/labels', 'r') as infile:
            reader = csv.reader(infile)
            for line in reader:
                y.append(int(line[0]))
            y = (np.array(y) - 1).reshape((-1, 1)).squeeze()
        
        cluster=10000
        test_size=10000
        
        np.random.seed(456)
        indices = np.arange(len(y))
        p = np.random.permutation(indices)
        x,y=x[indices],y[indices]
        whole_set=[(x[i],y[i]) for i in range(len(y))]

        
    elif dataset == 'location':
        x, y = [], []
        with open('data/location', 'r') as infile:
            reader = csv.reader(infile)
            for line in reader:
                y.append(int(line[0]))
                x.append([int(x) for x in line[1:]])
            x = np.array(x).astype(np.float32)
            y = (np.array(y) - 1).reshape((-1, 1)).squeeze()
        cluster=1600
        test_size=1600
        np.random.seed(456)
        indices = np.arange(len(y))
        p = np.random.permutation(indices)
        x,y=x[indices],y[indices]
        whole_set=[(x[i],y[i]) for i in range(len(y))]
        

    length  = len(whole_set)
    
    if mode == 'target':
        train_size = cluster
        remain_size = length - train_size - test_size
        train_set, _, test_set = dataset_split(whole_set, [train_size, remain_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'shadow': 
        train_size = cluster
        remain_size = length - train_size - test_size
        train_set,_, test_set = dataset_split(whole_set, [train_size, remain_size, test_size], seed=123)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'salem_unknown': 
        train_size = length - max_cluster - test_size
        salme_train = int(train_size * 0.5)
        salme_test = train_size - salme_train
        _, train_set, test_set, _ = dataset_split(whole_set, [max_cluster, salme_train, salme_test, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'salem_known': 
        salme_train = cluster
        salme_test = cluster
        rest_size =  length - max_cluster - test_size - cluster - cluster
        _, train_set, test_set, _, _ = dataset_split(whole_set, [max_cluster, salme_train, salme_test, rest_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader, test_loader
    elif mode == 'ChangeDataSize':
        train_size = cluster
        remain_size = length - max_cluster - train_size - test_size
        _, train_set, _, _ = dataset_split(whole_set, [max_cluster, train_size, remain_size, test_size])
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        return train_loader

    ###
    elif mode in ['adversary', 'radius', 'adversary_shadow']:
        if mode=='adversary_shadow':
            s=123
        else:
            s=1

        mem_size = min([cluster, test_size, max_num])

        non_size = mem_size
        remain_size = length - mem_size - non_size
        mem_set, _, non_set = dataset_split(whole_set, [mem_size, remain_size, non_size],seed=s)
        print('member set size is :   ',len(mem_set), '  mode is : ' +mode)
        data_set = ConcatDataset([mem_set, non_set])
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False, **kwargs)
        if mode == 'radius':
            return mem_set, non_set, transform
        else:
            return data_loader
     


def load_dataset_DataAug_AdvReg(args, dataset, cluster=None, defense=None):
    kwargs = {'num_workers': 2, 'pin_memory': True}
    # load trainset and testset
    if defense == 'DataAug':
        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]) # transforms.RandomErasing();  Rand_Augment()
    elif defense == 'AdvReg':
        transform = transforms.Compose([transforms.ToTensor()])

    if dataset == 'CIFAR10':
        whole_set = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        test_size = 1000

    length = len(whole_set)
    
    train_size = cluster
    remain_size = length - train_size - test_size


    train_set, _, _ = dataset_split(whole_set, [train_size, remain_size, test_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader

def dataset_split(dataset, lengths,seed=1):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = list(range(sum(lengths)))
    np.random.seed(seed)
    np.random.shuffle(indices)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def calc_confuse(preds, labels):
    labels = labels.astype(np.bool).squeeze()
    preds = np.argmax(preds, axis=1).astype(np.bool)
    tp = np.logical_and(np.equal(labels, True), np.equal(preds, True)).astype(np.int).sum()
    fp = np.logical_and(np.equal(labels, False), np.equal(preds, True)).astype(np.int).sum()
    tn = np.logical_and(np.equal(labels, False), np.equal(preds, False)).astype(np.int).sum()
    fn = np.logical_and(np.equal(labels, True), np.equal(preds, False)).astype(np.int).sum()
    N=float(len(labels))
    return tp/N, fp/N, tn/N, fn/N
        
def train_attack_model(args,train_set, test_set, type_):
    
    import torch.optim as optim
    
    model = AttackModel(input_size=train_set[0].shape[1],aug_type=type_)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    

    tensor_x = torch.Tensor(list(train_set[0])) # transform to torch tensor
    tensor_y = torch.Tensor(list(train_set[1]))
    train_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = DataLoader(train_dataset) # create your dataloader

    tensor_x_test = torch.Tensor(list(test_set[0])) # transform to torch tensor
    tensor_y_test = torch.Tensor(list(test_set[1]))
    test_dataset = TensorDataset(tensor_x_test,tensor_y_test) # create your datset
    test_loader = DataLoader(test_dataset) # create your dataloader
    
    for epoch in range(1, 51):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            print(target, data)
            target=target.type(torch.LongTensor)
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            
        

        model.eval()
        test_loss=0
        correct=0
        with torch.no_grad():
            for data, target in test_loader:
                target=target.type(torch.LongTensor)
                data, target = data.cuda(), target.cuda()
                output = model(data)
                test_loss += F.cross_entropy(output, target).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()   

        test_loss /= len(test_loader.dataset)

        accuracy = 100. * correct / len(test_loader.dataset)
        print('\nDiscriminator: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy))

    pred_train = model(tensor_x.cuda())
    pred_test= model(tensor_x_test.cuda())
    tp_train, fp_train, tn_train, fn_train = calc_confuse(pred_train.detach().cpu().numpy(), train_set[1])
    tp_test, fp_test, tn_test, fn_test = calc_confuse(pred_test.detach().cpu().numpy(), test_set[1])

    # [tp_train, fp_train, tn_train, fn_train], [tp_test, fp_test, tn_test, fn_test]
    return (tp_train+ tn_train) , (tp_test+tn_test)


class AttackModel(nn.Module):
    def __init__(self,input_size, aug_type='n'):
        """ Sample Attack Model.
        :param aug_type:
        """
        self.input_size=input_size
        super().__init__()
        if aug_type == 'n':
            self.classifier = nn.Sequential(
                nn.Linear(input_size, 10),
                nn.ReLU(True),
                nn.Linear(10, 2) )

        elif aug_type == 'r' or aug_type == 'd':
            self.classifier = nn.Sequential(
                nn.Linear(input_size, 10),
                nn.ReLU(True),
                nn.Linear(10, 10),
                nn.ReLU(True),
                nn.Linear(10, 2) )
            
        else:
            raise ValueError(f"aug_type={aug_type} is not valid.")

    def forward(self, x):
        out = self.classifier(x)
        return out   

def get_threshold(source_m, source_stats, target_m, target_stats):
    """ Train a threshold attack model and get teh accuracy on source and target models.
        Args:
        source_m: membership labels for source dataset (1 for member, 0 for non-member)
        source_stats: scalar values to threshold (attack features) for source dataset
        target_m: membership labels for target dataset (1 for member, 0 for non-member)
        target_stats: scalar values to threshold (attack features) for target dataset
        Returns: best acc from source thresh, precision @ same threshold, threshold for best acc,
        precision at the best threshold for precision. all tuned on source model.
        """
        # find best threshold on source data
    acc_source, t, prec_source, tprec = get_max_accuracy(source_m, source_stats)

    # find best accuracy on test data (just to check how much we overfit)
    acc_test, _, prec_test, _ = get_max_accuracy(target_m, target_stats)

    # get the test accuracy at the threshold selected on the source data
    acc_test_t, _, _, _ = get_max_accuracy(target_m, target_stats, thresholds=[t])
    _, _, prec_test_t, _ = get_max_accuracy(target_m, target_stats, thresholds=[tprec])
    print("acc src: {}, acc test (best thresh): {}, acc test (src thresh): {}, thresh: {}".format(acc_source, acc_test,
                                                                                                acc_test_t, t))
    print("prec src: {}, prec test (best thresh): {}, prec test (src thresh): {}, thresh: {}".format(prec_source, prec_test,
                                                                                               prec_test_t, tprec))     

    return acc_source, acc_test, acc_test_t

def get_max_accuracy(y_true, probs, thresholds=None):
    """Return the max accuracy possible given the correct labels and guesses. Will try all thresholds unless passed.
        Args:
        y_true: True label of `in' or `out' (member or non-member, 1/0)
        probs: The scalar to threshold
        thresholds: In a blackbox setup with a shadow/source model, the threshold obtained by the source model can be passed
        here for attackin the target model. This threshold will then be used.
        Returns: max accuracy possible, accuracy at the threshold passed (if one was passed), the max precision possible,
        and the precision at the threshold passed.
    """
    if thresholds is None:
        fpr, tpr, thresholds = roc_curve(y_true, probs)

    accuracy_scores = []
    precision_scores = []
    for thresh in thresholds:
        accuracy_scores.append(accuracy_score(y_true,[1 if m > thresh else 0 for m in probs]))
        precision_scores.append(precision_score(y_true, [1 if m > thresh else 0 for m in probs]))

    accuracies = np.array(accuracy_scores)
    precisions = np.array(precision_scores)
    max_accuracy = accuracies.max()
    max_precision = precisions.max()
    max_accuracy_threshold = thresholds[accuracies.argmax()]
    max_precision_threshold = thresholds[precisions.argmax()]



    return max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold


class Rand_Augment():
    def __init__(self, Numbers=None, max_Magnitude=None):
        # self.transforms = ['autocontrast', 'equalize', 'rotate', 'solarize', 'color', 'posterize',
        #                    'contrast', 'brightness', 'sharpness', 'shearX', 'shearY', 'translateX', 'translateY']

        self.transforms = ['rotate', 'shearX', 'shearY', 'translateX', 'translateY']
        if Numbers is None:
            self.Numbers = len(self.transforms) *10
        else:
            self.Numbers = Numbers
        if max_Magnitude is None:
            self.max_Magnitude = 10
        else:
            self.max_Magnitude = max_Magnitude
        fillcolor = 128
        self.ranges = {
            # these  Magnitude   range , you  must test  it  yourself , see  what  will happen  after these  operation ,
            # it is no  need to obey  the value  in  autoaugment.py
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 0.2, 10),
            "translateY": np.linspace(0, 0.2, 10),
            "rotate": np.linspace(0, 360, 10),
            # "color": np.linspace(0.0, 0.9, 10),
            # "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            # "solarize": np.linspace(256, 231, 10),
            # "contrast": np.linspace(0.0, 0.5, 10),
            # "sharpness": np.linspace(0.0, 0.9, 10),
            # "brightness": np.linspace(0.0, 0.3, 10),
            # "autocontrast": [0] * 10,
            # "equalize": [0] * 10,           
            # "invert": [0] * 10
        }
        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fill=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fill=fillcolor),
            "rotate": lambda img, magnitude: self.rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            # "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            # "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            # "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            # "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
            #     1 + magnitude * random.choice([-1, 1])),
            # "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
            #     1 + magnitude * random.choice([-1, 1])),
            # "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
            #     1 + magnitude * random.choice([-1, 1])),
            # "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            # "equalize": lambda img, magnitude: img,
            # "invert": lambda img, magnitude: ImageOps.invert(img)
        }

    def rand_augment(self):
        """Generate a set of distortions.
             Args:
             N: Number of augmentation transformations to apply sequentially. N  is len(transforms)/2  will be best
             M: Max_Magnitude for all the transformations. should be  <= self.max_Magnitude """

        M = np.random.randint(0, self.max_Magnitude, self.Numbers)

        sampled_ops = np.random.choice(self.transforms, self.Numbers)
        return [(op, Magnitude) for (op, Magnitude) in zip(sampled_ops, M)]

    def __call__(self, image):
        operations = self.rand_augment()
        for (op_name, M) in operations:
            operation = self.func[op_name]
            mag = self.ranges[op_name][M]
            image = operation(image, mag)
        return image

    def rotate_with_fill(self, img, magnitude):
        #  I  don't know why  rotate  must change to RGBA , it is  copy  from Autoaugment - pytorch
        rot = img.convert("RGBA").rotate(magnitude)
        return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

def fixed_seed(args):
    if args.seed is not None:
            np.random.seed(args.seed)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            cudnn.deterministic = True
##############################################
##############################################
##############################################
import torch.nn.init as init
import numpy as np

init_param = np.sqrt(2)
init_type = 'default'
def init_func(m):
    classname = m.__class__.__name__
    if classname.startswith('Conv') or classname == 'Linear':
        if getattr(m, 'bias', None) is not None:
                init.constant_(m.bias, 0.0)
        if getattr(m, 'weight', None) is not None:
            if init_type == 'normal':
                init.normal_(m.weight, 0.0, init_param)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight, gain=init_param)
            elif init_type == 'xavier_unif':
                init.xavier_uniform_(m.weight, gain=init_param)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_in')
            elif init_type == 'kaiming_out':
                init.kaiming_normal_(m.weight, a=init_param, mode='fan_out')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight, gain=init_param)
            elif init_type == 'zero':
                init.zeros_(m.weight)
            elif init_type == 'one':
                init.ones_(m.weight)
            elif init_type == 'constant':
                init.constant_(m.weight, init_param)
            elif init_type == 'default':
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            m.weight.data.fill_(1)
        if getattr(m, 'bias', None) is not None:
            m.bias.data.zero_()

def save_code(path):
    os.makedirs(path + '/code', exist_ok=True)
    
    shutil.copyfile('main.py', path + '/code/main.py')
    shutil.copyfile('deeplearning.py', path + '/code/deeplearning.py')
    shutil.copyfile('classifier.py', path + '/code/classifier.py')
    shutil.copyfile('utils.py', path + '/code/utils.py')
    shutil.copyfile('plot.py', path + '/code/plot.py')
    shutil.copyfile('attack.py', path + '/code/attack.py')
