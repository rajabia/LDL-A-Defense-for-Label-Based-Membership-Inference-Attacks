import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import argparse
import time
import random
import time
import math
import numpy as np
from runx.logx import logx
import pandas as pd
import torch
import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from models import ResNet18
from classifier import CNN
from utils import load_dataset, init_func
from deeplearning import test_target_model, test_shadow_model

from torch.autograd import Variable
# pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
action = -1
def test_models(args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    args.batch_size = 1
    args.logdir = './results/'+ args.datasets[args.dataset_ID]
    print(args.datasets[args.dataset_ID]+'_FGS.npy',split_size)


    acc_train, acc_test=[],[]
    args.augmentation=0
    # cluster, (member, non-member),(defense Off, ON), eps
    # results= np.zeros((len(split_size),2,9))
    results= np.load(args.datasets[args.dataset_ID]+'_FGS.npy')
    results[:,1,:]=0
    for idx, cluster in enumerate(split_size):
        torch.cuda.empty_cache() 
        logx.initialize(logdir=args.logdir + '/target/' + str(cluster), coolname=False, tensorboard=False)
        train_loader, test_loader = load_dataset(args, dataset, cluster, mode='target')
        targetmodel = CNN('CNN7', dataset,True,0.02)
        targetmodel = nn.DataParallel(targetmodel.cuda())
        state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
        targetmodel.load_state_dict(state_dict)
        targetmodel.eval()

        count=0
        acc_train=0
        acc_test=0
        # advs={'adv':[], 'y':[]}
        ads = np.load('cluster_adv_'+str(cluster)+'.npy',allow_pickle=True)
        ads=ads.item()

        for i in range(len(ads['y'])):
            x_adversarial,y=torch.from_numpy(ads['adv'][i]).cuda(), torch.from_numpy(ads['y'][i]).cuda()
            count+=1
            # x_adversarial=FGSAttack(targetmodel,data, y)
            print(x_adversarial.shape)
            pred=[]
            for j in range(x_adversarial.shape[0]):
                pred.append(targetmodel(x_adversarial[j:j+1]).detach().cpu().numpy())
            pred= np.argmax(np.array(pred), axis=1)
            corrects=[1 if pred[i] ==y.item() else 0 for i in range(len(pred))  ]
            results[idx, 1,:]=results[idx,1,:]+corrects
        print(results[idx,0,:])
        results[idx,:,:]=results[idx,:,:]/float(len(ads['y']))
        print(results[idx,0,:])



    np.save(args.datasets[args.dataset_ID]+'_FGS.npy',results)

def train_fgs(args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    args.batch_size = 1
    args.logdir = './results/'+ args.datasets[args.dataset_ID]


    acc_train, acc_test=[],[]
    args.augmentation=0
    # cluster, (member, non-member),(defense Off, ON), eps
    # 4 : defense free, LDL-0.02, LDL-0.04, LDL-0.06 ==> results: np.zeros((len(split_size),4,9))

    for idx, cluster in enumerate(split_size):
        torch.cuda.empty_cache() 
        logx.initialize(logdir=args.logdir + '/target/' + str(cluster), coolname=False, tensorboard=False)
        train_loader, test_loader = load_dataset(args, dataset, cluster, mode='target')
        targetmodel = CNN('CNN7', dataset,False,0)
        targetmodel = targetmodel.cuda()
        state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
        targetmodel.load_state_dict(state_dict)
        targetmodel.eval()
        test_loader_2=[]
        if mode=='members':
            for x,y in train_loader:
                pred= np.argmax(targetmodel(x.cuda()).detach().cpu().numpy())
                if pred==y:
                    test_loader_2.append((x.cuda(),y))
                if len(test_loader_2)>=100:
                    break
        else:
            for x,y in test_loader:
                pred= np.argmax(targetmodel(x.cuda()).detach().cpu().numpy())
                if pred==y:
                    test_loader_2.append((x.cuda(),y))
                if len(test_loader_2)>=100:
                    break

        print('length of correctly classified is ', len(test_loader_2))

        advs , y_true =  [], []
        for x,y in test_loader_2:
            adv_imgs=FGSAttack(targetmodel,x, y)
            advs.append(adv_imgs.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())

        if mode=='members':
            np.save('FGS/members'+dataset+'_'+str(idx)+'_FGS_data.npy', {'x': np.array(advs), 'y':y_true})
        else:
            np.save('FGS/'+dataset+'_'+str(idx)+'_FGS_data.npy', {'x': np.array(advs), 'y':y_true})

def eval_fgs(sigma,idx, cluster, args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    args.batch_size = 1
    args.logdir = './results/'+ args.datasets[args.dataset_ID]



    acc_train, acc_test=[],[]
    args.augmentation=0


    results=[]

    if mode=='members':
        data=np.load('FGS/members'+dataset+'_'+str(idx)+'_FGS_data.npy', allow_pickle=True)
    else:
        data=np.load('FGS/'+dataset+'_'+str(idx)+'_FGS_data.npy', allow_pickle=True)
    data=data.item()
    x=data['x']
    y=data['y']


    torch.cuda.empty_cache() 
    logx.initialize(logdir=args.logdir + '/target/' + str(cluster), coolname=False, tensorboard=False)
    if sigma>0:
        targetmodel = CNN('CNN7', dataset,True,sigma)
    else:
         targetmodel = CNN('CNN7', dataset,False,sigma)

    targetmodel =targetmodel.cuda()
    state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
    targetmodel.load_state_dict(state_dict)
    targetmodel.eval()
    N=min(100, x.shape[0])
    temp=np.zeros((N,x.shape[1]))

    for k in range(N):
        for t in range(x.shape[1]):
            preds= np.argmax(targetmodel(torch.from_numpy(x[k,t:t+1,:,:,:]).cuda()).detach().cpu().numpy())
            if preds==y[k] :
                temp[k,t]= 1 
            else:
                temp[k,t]=0 


    torch.cuda.empty_cache() 
    return np.sum(temp, axis=0)/N





def FGSAttack(model,image_tensor, y_true):
    eps_list=[0.001, 0.005, 0.01, 0.015, 0.02, 0.04, 0.06 , 0.08, 0.1]
    adv_imgs=[]

    img_variable = Variable(image_tensor, requires_grad=True)
    target = Variable(torch.LongTensor([y_true]), requires_grad=False).cuda()
    output=model(img_variable.cuda())
    loss = torch.nn.CrossEntropyLoss()
    loss_cal = loss(output, target)
    loss_cal.backward(retain_graph=True) 
    x_grad = torch.sign(img_variable.grad.data)
    for eps in eps_list:				
        x_adversarial = img_variable.data + eps * x_grad		  
        adv_imgs.append(x_adversarial)  
    adv_imgs=torch.squeeze(torch.stack(adv_imgs,axis=0))
    return adv_imgs 




##############################
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Decision-based Membership Inference Attack Toy Example') 
    parser.add_argument('--train', default=True, type=bool,
                        help='train or attack')
    parser.add_argument('--dataset_ID', default=0, type=int, 
                        help='CIFAR10=0, CIFAR100=1, GTSRB=2, Face=3')
    parser.add_argument('--datasets', nargs='+',
                        default=['CIFAR10', 'CIFAR100', 'GTSRB', 'Face'])
    parser.add_argument('--num_classes', nargs='+',
                        default=[10, 100, 43, 19])
    parser.add_argument('--Split-Size', nargs='+',
                        default=[[ 3000],  # , 2000, 1500, 1000,500, 100
                                [15000, 25000, 35000, 40000 ],
                                [600, 500, 400, 300, 200,100 ],  #600, 500, 400, 300, 200, 100			
                                [1400, 1000, 700, 300  ],  #350, 300, 250, 200, 150, 100				
                                ]) 
    parser.add_argument('--batch-size', nargs='+', default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--cuda', default=True,type=bool,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--logdir', type=str, default='',
                        help='target log directory')
    parser.add_argument('--mode_type', type=str, default='',
                        help='the type of action referring to the load dataset')

    parser.add_argument('--defense', type=int)



    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('***'*10, device, '***'*10 )




    train_fgs(args)
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    results= np.zeros((len(split_size),4,9))
 
    # 4 : Naive, LDL-0.02, LDL-0.04, LDL-0.06

    LDL_sigmas=[0,0.02, 0.04,0.06]
    train_fgs(args)
    for j, sigma in enumerate(LDL_sigmas):
        torch.cuda.empty_cache() 
        print('*'*10+'   Running for   ', sigma)
        for idx, cluster in enumerate(split_size):
            temp=eval_fgs(sigma,idx, cluster,args)
            results[idx,j,:]=temp
            print(temp)
        print('Saving To ', mode+dataset+'_FGS.npy')
        if mode=='members':
            np.save('FGS/members'+dataset+'_FGS.npy',results)
        else:
            np.save('FGS/nonmembers_'+dataset+'_FGS.npy',results)



if __name__ == "__main__":
    mode='nonmembers'
    if not os.path.exists('./FGS'):
        os.mkdir('./FGS')
    for mode in ['members', 'nonmembers']
        main()
