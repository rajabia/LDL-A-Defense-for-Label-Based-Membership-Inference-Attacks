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
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from classifier import CNN, fc_connect
from utils import load_dataset, init_func, Rand_Augment
from deeplearning import train_target_model, test_target_model, train_shadow_model, test_shadow_model
from attack import binary_rand_robust,icml_attack, AdversaryOne_Feature, AdversaryOne_evaluation, AdversaryTwo_HopSkipJump
#
from opacus import PrivacyEngine
# pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#/home/nsl/anaconda3/envs/LDL-env/lib/python3.10/site-packages/art/estimators/classification/classifier.py
action = -1


def test_models_with_defense(args,modelpath,description='',params=[0,1],targetmodel=None):
    
    acc_train, acc_test=0,0
    args.augmentation=0
    logx.initialize(logdir=args.logdir + '/target/' + str(2000), coolname=False, tensorboard=False)
    torch.cuda.empty_cache() 
    dataset='CIFAR10'
    print(modelpath)
    train_loader, test_loader = load_dataset(args, dataset, 2000, mode='target')
    if targetmodel==None:
        if 'dropout' in description:
            targetmodel = CNN('CNN7', dataset,False,args.sigma,dropout=params[0])
        else:
            targetmodel = CNN('CNN7', dataset,args.defense,args.sigma)
        train_loader, test_loader = load_dataset(args, dataset, 2000, mode='target')
        if 'dp' in description:
            optimizer=optim.SGD(targetmodel.parameters(), lr=args.lr)
            privacy_engine = PrivacyEngine(secure_mode=False)
            targetmodel, optimizer, train_loader = privacy_engine.make_private_with_epsilon(module=targetmodel,
                optimizer=optimizer,data_loader=train_loader, max_grad_norm=2.0, epochs=100,
                target_epsilon=params[0], target_delta=1./2000)


        targetmodel = targetmodel.cuda()
        state_dict, _ =  logx.load_model(path=modelpath)
        targetmodel.load_state_dict(state_dict)
    

    targetmodel.eval()
    acc_test=test_target_model(args, targetmodel, test_loader, 100, save=False)
    acc_train=test_target_model(args, targetmodel, train_loader, 100, save=False)
    return acc_train,acc_test

def test_models(args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    if  args.defense:
        
        pathadd=args.logdir + '/target/accuracy_' + str(args.dataset_ID)+'.npy'
    else:
        
        pathadd=args.logdir + '/target/NoDefenseaccuracy_' + str(args.dataset_ID)+'.npy'
    print(pathadd,split_size)


    acc_train, acc_test=[],[]
    args.augmentation=0
    for idx, cluster in enumerate(split_size):
        torch.cuda.empty_cache() 
        logx.initialize(logdir=args.logdir + '/target/' + str(cluster), coolname=False, tensorboard=False)
        train_loader, test_loader = load_dataset(args, dataset, cluster, mode='target')
        targetmodel = CNN('CNN7', dataset,args.defense,args.sigma)
        if args.dataset_ID in [4,5,6]:
            targetmodel = fc_connect( dataset,args.defense,args.sigma)
        targetmodel = targetmodel.cuda()

        print(args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
        state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
        targetmodel.load_state_dict(state_dict)
        # targetmodel.eval()
        

        acc_test.append(test_target_model(args, targetmodel, test_loader, 100, save=False))
        acc_train.append(test_target_model(args, targetmodel, train_loader, 100, save=False))
    

def change_Sigma(args):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    args.defense=True
    args.augmentation=0
    
    pathadd=args.logdir + '/target/sigma_' + str(args.dataset_ID)+'.npy'
    sigmas=[ 0.008, 0.012, 0.015, 0.02, 0.03, 0.04]
    acc_train, acc_test=np.zeros((6,len( args.Split_Size[args.dataset_ID]))),np.zeros((6,len( args.Split_Size[args.dataset_ID])))
    print(split_size)
    for i,cluster in  enumerate(split_size):
        
        
        
        for idx, sigma in enumerate(sigmas):
            torch.cuda.empty_cache() 
            logx.initialize(logdir=args.logdir + '/target/' +str(cluster) , coolname=False, tensorboard=False)
            train_loader, test_loader = load_dataset(args, dataset, cluster, mode='target')
            targetmodel = CNN('CNN7', dataset,args.defense, sigma)
            targetmodel = targetmodel.cuda()

        
            state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
            targetmodel.load_state_dict(state_dict)
            targetmodel.eval()
            data_loader = load_dataset(args, dataset, cluster, mode='adversary', max_num=100)

            acc_test[idx,i]=test_target_model(args, targetmodel, test_loader, 100, save=False)
            acc_train[idx,i]=test_target_model(args, targetmodel, train_loader, 100, save=False)
            print( 'dataset:   ',args.dataset_ID, '   sigma:   ',sigma, '   accuracies:  ',      acc_test[idx,i],acc_train[idx,i])
	
    print(pathadd)
    np.save(pathadd,{'sigma':sigmas , 'train_acc': acc_train, 'test_acc':acc_test})


def Train_Target_Model(args, folder='target', defense='none', params=[2,0.05]):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    print('Training Data set ', dataset,torch.cuda.device_count())
    for idx, cluster in enumerate(split_size):
        torch.cuda.empty_cache() 
        if defense== 'none':
            address=args.logdir + '/'+folder+'/'+ str(cluster)
        else:
            address=args.logdir + '/'+folder+'/'+defense+'_'+str(params[0])+ str()

        logx.initialize(logdir= address, coolname=False, tensorboard=False)
        train_loader, test_loader = load_dataset(args, dataset, cluster, mode=args.mode_type)
        if defense =='dropout' :
            targetmodel = CNN('CNN7', dataset,False,args.sigma,dropout=params[0])
        else:
            targetmodel = CNN('CNN7', dataset,False,args.sigma)
        if args.dataset_ID in [4,5,6]:
            targetmodel = fc_connect( dataset,False,args.sigma)
        
        targetmodel.apply(init_func)
        
        targetmodel = targetmodel.cuda()
        if defense== 'dp':
            optimizer=optim.SGD(targetmodel.parameters(), lr=args.lr)
            privacy_engine = PrivacyEngine(secure_mode=False)
            targetmodel, optimizer, train_loader = privacy_engine.make_private_with_epsilon(module=targetmodel,
                optimizer=optimizer,data_loader=train_loader, max_grad_norm=params[1], epochs=args.epochs,
                target_epsilon=params[0], target_delta=1./cluster)
        else:

            optimizer = optim.Adam(targetmodel.parameters(), lr=args.lr)
        if os.path.exists(address+'/best_checkpoint_ep.pth'):
            print('The model is already trained abd saved in '+address)
            state_dict, _ =  logx.load_model(path=address+'/best_checkpoint_ep.pth')
            targetmodel.load_state_dict(state_dict)
            targetmodel = targetmodel.cuda()
        else:
            logx.msg('======================Train_Target_Model {} ===================='.format(cluster))
            for epoch in range(1, args.epochs + 1):
                train_target_model(args, targetmodel, train_loader, optimizer, epoch,defense=defense, params=params)
                test_target_model(args, targetmodel, test_loader, epoch, save=True)
    return targetmodel



def AdversaryTwo(args, folder='target',path_model='none', description='',targetmodel=None,params=None):
    
    '''
        path_model: path address for miodels trained with a defense approach much should be declared otherwise 
        the function loads models trained without any defense
        description: description will be added to the name of files saving the final results
    '''

    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    num_class = args.num_classes[args.dataset_ID]
    
    logx.initialize(logdir=args.logdir + '/adversaryTwo', coolname=False, tensorboard=False)
    ITER = [150] # for call HSJA evaluation [1, 5, 10, 15, 20, 30]  default 50
    
    
    for maxitr in ITER:
        AUC_Dist, Distance = [], []
        for cluster in split_size:
            
            torch.cuda.empty_cache()
            if folder=='shadow':
                extension='shadow_'
            else: 
                extension=''
            if args.label_knowledge:
                prifix= ''+description
            else:
                prifix= 'WeakAdv_'+description
            if args.defense:
                add_auc=args.logdir + '/adversaryTwo/AUC_Dist_'+extension+prifix+args.blackadvattack + '.csv'
                add_dist=args.logdir + '/adversaryTwo/Distance_'+extension+prifix+args.blackadvattack+'.csv'
            else:
                add_auc=args.logdir + '/adversaryTwo/NoLDLDefenseAUC_Dist_'+extension+prifix+args.blackadvattack + '.csv'
                add_dist=args.logdir + '/adversaryTwo/NoLDLDefenseDistance_'+extension+prifix+args.blackadvattack+'.csv'
            args.batch_size=1
            if folder=='shadow':
                data_loader = load_dataset(args, dataset, cluster, mode='adversary_shadow', max_num=100)
            else:
                data_loader = load_dataset(args, dataset, cluster, mode='adversary', max_num=100)

            
            
            
            if not os.path.exists(add_auc):
                if targetmodel==None:
                    targetmodel = CNN('CNN7', dataset,args.defense,args.sigma)

                    if 'dropout' in description:
                        targetmodel = CNN('CNN7', dataset,False,args.sigma,dropout=params_def[0])
                    else:
                        targetmodel = CNN('CNN7', dataset,args.defense,args.sigma)
                    if path_model =='none':
                        path_model=args.logdir + '/'+folder+'/' + str(cluster) + '/best_checkpoint_ep.pth'

                    state_dict, _ =  logx.load_model(path=path_model)
                    targetmodel.load_state_dict(state_dict)

                targetmodel = targetmodel.cuda()
                targetmodel.eval()

                AUC_Dist, Distance = AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, 
                                                              cluster, AUC_Dist, Distance, False, maxitr)
                df = pd.DataFrame()
                AUC_Dist = pd.concat([df,pd.DataFrame.from_records(AUC_Dist)])
                Distance = pd.concat([df,pd.DataFrame.from_records(Distance)])
                AUC_Dist.to_csv(add_auc)
                Distance.to_csv(add_dist)
            else:
                print('The csv file for this expriment esxists at %s'%add_auc)
                AUC_Dist= pd.read_csv(add_auc)
                Distance = pd.read_csv(add_dist)
                Distance['L0Distance']=[s.replace('[', '') for s in Distance['L0Distance']]
                Distance['L0Distance']=[s.replace(']', '') for s in Distance['L0Distance']]
                Distance['L0Distance']=np.array(Distance['L0Distance']).astype(float)
                Distance['L1Distance']=[s.replace('[', '') for s in Distance['L1Distance']]
                Distance['L1Distance']=[s.replace(']', '') for s in Distance['L1Distance']]
                Distance['L1Distance']=np.array(Distance['L1Distance']).astype(float)
                Distance['L2Distance']=[s.replace('[', '') for s in Distance['L2Distance']]
                Distance['L2Distance']=[s.replace(']', '') for s in Distance['L2Distance']]
                Distance['L2Distance']=np.array(Distance['L2Distance']).astype(float)
        
        
        return AUC_Dist,Distance



def AdversaryICML(args, attack='r'):
    split_size = args.Split_Size[args.dataset_ID]
    dataset = args.datasets[args.dataset_ID]
    num_class = args.num_classes[args.dataset_ID]
    
    logx.initialize(logdir=args.logdir + '/adversaryTwo', coolname=False, tensorboard=False)
    ITER = [150] # for call HSJA evaluation [1, 5, 10, 15, 20, 30]  default 50

    aug_kwarg = args.d if attack == 'd' else args.r

    ### Data augmentation should be of for data loading
    args.augmentation=0
    
    AUC_Dist=[]
    for maxitr in ITER:
        AUC_Dist, Distance = [], []
        print(args.logdir + '/'+str(args.sigma)+'/'+attack + '.csv')
        for cluster in split_size:
            torch.cuda.empty_cache()

            args.batch_size=1
            
            train_loader_target, test_loader_target = load_dataset(args, dataset, cluster, mode='target', max_num=100)
            
            targetmodel = CNN('CNN7', dataset,args.defense,args.sigma)
            
            targetmodel = targetmodel.cuda()
            
            state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(cluster) + '/best_checkpoint_ep.pth')
            targetmodel.load_state_dict(state_dict)
            targetmodel.eval()
            

            train_loader_shadow, test_loader_shadow = load_dataset(args, dataset, cluster, mode='shadow', max_num=100)
            shadowmodel = CNN('CNN7', dataset,args.defense,args.sigma)
            shadowmodel = shadowmodel.cuda()
            
            if attack=='g':
                shadowmodel=None
            else:
                state_dict, _ =  logx.load_model(path=args.logdir + '/shadow/' + str(cluster) + '/best_checkpoint_ep.pth')
                shadowmodel.load_state_dict(state_dict)
                shadowmodel.eval()

            AUC_Dist=icml_attack(targetmodel,train_loader_target, test_loader_target, shadowmodel, train_loader_shadow, test_loader_shadow,AUC_Dist,cluster, attack,aug_kwarg,args)
            

        df = pd.DataFrame()
        AUC_Dist = pd.concat([df,pd.DataFrame.from_records(AUC_Dist)])

        if args.label_knowledge:
            prifix= ''
        else:
            prifix= 'WeakAdv_'
        
        if args.defense:
            
            AUC_Dist.to_csv(args.logdir + '/'+str(args.sigma)+'/'+prifix+attack + '_new.csv')
            
    
        else:
            AUC_Dist.to_csv(args.logdir+ '/'+str(args.sigma)+'/NoDefense'+prifix+attack+ '_new.csv')
                


def binary_attacks(args):
    from sklearn import metrics
    clusters=[10000,10000,1600]
    class_nb=[100,100,30]
    dims=[600,6168,446]
    input_dim=dims[args.dataset_ID-4]
    sigmas = np.array([1.,2.,3.,5.,10.])/ input_dim #1.,3.,5.,6.,8.,10.,
    dict_res={'sigma':sigmas, 'AUC':[], 'Acc':[]}
    dataset=args.datasets[args.dataset_ID]
    max_samples = min(10000, clusters[args.dataset_ID-4])
    loader_target= load_dataset(args, dataset, None, mode='adversary', max_num=max_samples)
    targetmodel = fc_connect(dataset,args.defense,args.sigma)
    targetmodel =targetmodel.cuda()

    print(args.logdir + '/target/' + str(clusters[args.dataset_ID-4]) + '/best_checkpoint_ep.pth')
    state_dict, _ =  logx.load_model(path=args.logdir + '/target/' + str(clusters[args.dataset_ID-4]) + '/best_checkpoint_ep.pth')
    targetmodel.load_state_dict(state_dict)

    targetmodel.eval()

    # train_loader_shadow, test_loader_shadow= load_dataset(args, dataset, cluster, mode='shadow', max_num=100)
    # targetmodel = fc_connect( dataset,args.defense,args.sigma)
    # state_dict, _ =  logx.load_model(path=args.logdir + '/shadow/' + str(cluster) + '/best_checkpoint_ep.pth')
    # targetmodel.load_state_dict(state_dict)
    # targetmodel.eval()

    for sigma in sigmas:
        print('%'*20, sigma)
        noise_target=binary_rand_robust(args,targetmodel, loader_target, sigma, noise_samples=10000)

        noise_target_in=noise_target[:int(len(noise_target)/2)]
        noise_target_out=noise_target[int(len(noise_target)/2):]
        
        auc=metrics.roc_auc_score([0]*len(noise_target_in)+[1]*len(noise_target_out), noise_target_in+noise_target_out)
        dict_res['AUC'].append(auc)

        fpr, tpr, thresholds=metrics.roc_curve([0]*len(noise_target_in)+[1]*len(noise_target_out), noise_target_in+noise_target_out)
        max_acc=[]
        for thresh in thresholds+[0]:
            max_acc.append(metrics.accuracy_score([0]*len(noise_target_in)+[1]*len(noise_target_out),[0 if m > thresh else 1 for m in noise_target]))

        dict_res['Acc'].append(np.array(max_acc).max())
        if args.defense:
            np.save(args.logdir +'/'+str(args.sigma)+'/b_'+str(args.dataset_ID)+'_cl_'+str(clusters[args.dataset_ID-4])+'.npy',dict_res)  
        else:
            np.save(args.logdir +'/'+str(args.sigma)+'/NoDefense_b_'+str(args.dataset_ID)+'_cl_'+str(clusters[args.dataset_ID-4])+'.npy',dict_res)
        print(dict_res)


def retraining_defenses(args):
    defenses=['dp','dropout', 'L1', 'L2']
    cluster= args.Split_Size[args.dataset_ID][0]

def retraining_defenses(args):
    defenses=['dp','dropout', 'L1', 'L2']
    cluster= args.Split_Size[args.dataset_ID][0]
    
    args.epochs=100
    args.batch_size=128
    
    def MIA_asr(distances):
        
        from sklearn.metrics import accuracy_score,roc_curve
        labels=[0]*int(len(distances)/2)+[1]*int(len(distances)/2)
        fpr, tpr, thresholds = roc_curve(labels, distances, pos_label=2)
        max_acc=0
        for t in thresholds:
            y_pred=[0 if d>t else 1for d in distances]
            acc=accuracy_score(labels, y_pred)
            max_acc=max(max_acc, acc)
        return max_acc
    
    text='defense\t param\t ACCTrain\t ACCtest\t L0Distance\t L1Distance\t L2Distance\n'
    for l2 in [0.00001]:#0.0001,0.001,0.01,0.1,1,10,100
        print('Defense of L2, coefficient value of : ',l2)
        address=args.logdir + '/target/L2_'+str(l2)+'/best_checkpoint_ep.pth'
        args.epochs=100
        args.batch_size=128
        targetmodel=Train_Target_Model(args, folder='target', defense='L2', params=[l2])
        acc_train,acc_test=test_models_with_defense(args,address)
        AUC_Dist,Distance= AdversaryTwo(args, folder='target',path_model=address, description='L2_'+str(l2)+'_')   
        a,b,c=MIA_asr(Distance['L0Distance']),MIA_asr(Distance['L1Distance']),MIA_asr(Distance['L2Distance'])
        text=text+ 'L2\t'+str(l2) +'\t'+str(acc_train)+'\t'+str(acc_test)+'\t'+str(a)+'\t'+str(b)+'\t'+str(c)+'\n'
    with open('cifarL2.csv', 'w') as f:
        f.write(text)
        
    text='defense\t param\t ACCTrain\t ACCtest\t L0Distance\t L1Distance\t L2Distance\n'
    for eps in [5,20,30,100]:#
        print('Defense of DP, epsilon value of : ', eps)
        args.epochs=100
        args.batch_size=128
        address=args.logdir + '/target/dp_'+str(eps)+'/best_checkpoint_ep.pth'
        targetmodel= Train_Target_Model(args, folder='target', defense='dp', params=[eps,2.0])
        
        acc_train,acc_test=test_models_with_defense(args,address,description='dp',params=[eps],targetmodel=targetmodel)
        AUC_Dist,Distance= AdversaryTwo(args, folder='target',path_model=address, description='dp_'+str(eps)+'_',params=[eps,2.0],targetmodel=targetmodel) 
        a,b,c=MIA_asr(Distance['L0Distance']),MIA_asr(Distance['L1Distance']),MIA_asr(Distance['L2Distance'])
        text=text+ 'DP\t'+str(eps) +'\t'+str(acc_train)+'\t'+str(acc_test)+'\t'+str(a)+'\t'+str(b)+'\t'+str(c)+'\n'
        with open('cifarDP.csv', 'w') as f:
            f.write(text)
    
    
    
    text='defense\t param\t ACCTrain\t ACCtest\t L0Distance\t L1Distance\t L2Distance\n'
    for l1 in [0.000001,0.00001, 0.0001,0.001,0.01,0.1,1,10,100]:
        args.epochs=100
        args.batch_size=128
        print('Defense of L1, coefficient value of : ',l1)
        address=args.logdir + '/target/L1_'+str(l1)+'/best_checkpoint_ep.pth'
        Train_Target_Model(args, folder='target', defense='L1', params=[l1])
        acc_train,acc_test=test_models_with_defense(args,address)
        AUC_Dist,Distance= AdversaryTwo(args, folder='target',path_model=address, description='L1_'+str(l1)+'_') 
        a,b,c=MIA_asr(Distance['L0Distance']),MIA_asr(Distance['L1Distance']),MIA_asr(Distance['L2Distance'])
        text=text+ 'L1\t'+str(l1) +'\t'+str(acc_train)+'\t'+str(acc_test)+'\t'+str(a)+'\t'+str(b)+'\t'+str(c)+'\n'
    with open('cifarL1.csv', 'w') as f:
        f.write(text)
        
    text='defense\t sigma^2\t ACCTrain(TPR)\t ACCtest(FPR)\t L0Distance\t L1Distance\t L2Distance\t ASR\n'
    for sigma in [0.01,0.02, 0.03, 0.05,0.06, 0.07,0.09,0.11,0.13,0.15,0.17, 0.19, 0.21, 0.23]:
        args.epochs=100
        args.batch_size=128
        print('Defense of LDL, sigma value of  : ',sigma)
        if sigma>0:
            args.defense=True
            args.sigma=sigma
        address=args.logdir + '/target/2000/best_checkpoint_ep.pth'
        Train_Target_Model(args, folder='target')
        acc_train,acc_test=test_models_with_defense(args,address)
        AUC_Dist,Distance= AdversaryTwo(args, folder='target')
        a,b,c=MIA_asr(Distance['L0Distance']),MIA_asr(Distance['L1Distance']),MIA_asr(Distance['L2Distance'])
        text=text+ 'LDL\t'+str(sigma) +'\t'+str(acc_train)+'\t'+str(acc_test)+'\t'+str(a)+'\t'+str(b)+'\t'+str(c)+ '\t'+str(np.max(np.array([a,b,c])))+'\n'
        with open('LDL.csv', 'w') as f:
            f.write(text)

    
        
    
        
    text='defense\t param\t ACCTrain\t ACCtest\t L0Distance\t L1Distance\t L2Distance\n'
    for drop in [0.1,0.5,0.8]:
        print('Defense of Dropout, drouout value of : ', drop)
        args.epochs=100
        args.batch_size=128
        address=args.logdir + '/target/dropout_'+str(drop)+'/best_checkpoint_ep.pth'
        Train_Target_Model(args, folder='target', defense='dropout', params=[drop])
        acc_train,acc_test=test_models_with_defense(args,address,description='dropout',params=[drop])
        AUC_Dist,Distance= AdversaryTwo(args, folder='target',path_model=address, description='dropout_'+str(drop)+'_')
        a,b,c=MIA_asr(Distance['L0Distance']),MIA_asr(Distance['L1Distance']),MIA_asr(Distance['L2Distance'])
        text=text+ 'dropout\t'+str(drop) +'\t'+str(acc_train)+'\t'+str(acc_test)+'\t'+str(a)+'\t'+str(b)+'\t'+str(c)+'\n'
    with open('cifarDrop.csv', 'w') as f:
        f.write(text)
    
    
    
    
    
##############################
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Decision-based Membership Inference Attack Toy Example') 
    parser.add_argument('--train', default=True, type=bool,
                        help='train or attack')
    parser.add_argument('--dataset_ID', default=-1, type=int, 
                        help='CIFAR10=0, CIFAR100=1, GTSRB=2, Face=3, purchase=4 , texas=5, location=6')
    parser.add_argument('--datasets', nargs='+',
                        default=['CIFAR10', 'CIFAR100', 'GTSRB', 'Face', 'purchase','texas', 'location'])
    parser.add_argument('--num_classes', nargs='+',
                        default=[10, 100, 43, 19])
    parser.add_argument('--Split-Size', nargs='+',
                        default=[[ 3000, 2000, 1500, 1000,500, 100],  # 
                                [15000, 25000, 35000, 40000 ], #[7000, 6000, 5000, 4000, 3000, 2000 ],                     #9000, 8000, 7000, 6000, 5000, 4000  # 7000, 6000, 5000, 4000, 3000, 2000
                                [600, 500, 400, 300, 200,100 ],  #600, 500, 400, 300, 200, 100            
                                [1400, 1000, 700, 300  ],  #350, 300, 250, 200, 150, 100                
                                [10000],[10000],[1600]]) 
    parser.add_argument('--batch-size', nargs='+', default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001 for adam; 0.1 for SGD)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--cuda', default=True,type=bool,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--blackadvattack', default='HopSkipJump', type=str,
                        help='adversaryTwo uses the adv attack the target Model: HopSkipJump; QEBA')
    parser.add_argument('--logdir', type=str, default='',
                        help='target log directory')
    parser.add_argument('--mode_type', type=str, default='',
                        help='the type of action referring to the load dataset')
    parser.add_argument('--advOne_metric', type=str, default='Loss_visual', help='AUC of Loss, Entropy, Maximum respectively; or Loss_visual')
    
    parser.add_argument('--defense', type=int,  default= 0)
    parser.add_argument('--label-knowledge', type=int,  default=1)
    parser.add_argument('--augmentation', type=int,  default=0)
    parser.add_argument('--action', type=int,  default= -1)

    parser.add_argument('--r', default=7, type=int, help='r param in rotation attack if used')
    parser.add_argument('--d', default=1, type=int, help='d param in translation attack if used')

    parser.add_argument('--sigma', default=0.02, type=float, help='d param in translation attack if used')

    parser.add_argument('--attack-type', type=str, default='g')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('***'*10, device, '***'*10 )
    
    
    
    if not args.augmentation:
        folder='results'+'/' 
    else:
        args.Split_Size = [[3000],[40000],[600], [1400]]
        folder='results_augmentation'+'/' 
 
    
    print('Label Knowledge: '+str(args.label_knowledge))
    print('defense: '+str(args.defense))
    print('augmentation: '+str(args.augmentation))
    print('action: '+str(args.action))
    if args.action == 5: 
        print('attack: '+str(args.attack_type))

    

    
    if args.augmentation:
        args.Split_Size[args.dataset_ID] = [args.Split_Size[args.dataset_ID][0]]
    for action in [args.action]:
            
        
        if action == 0:
            for dataset_idx in [args.dataset_ID]:
        
                args.dataset_ID = dataset_idx
                args.logdir = folder+ args.datasets[args.dataset_ID]
                print('*'*10, '   Learning Target Models   ', '*'*10)
                args.mode_type = 'target'
                print(args.Split_Size[args.dataset_ID])
                Train_Target_Model(args)
                # test_models(args)
        elif action == 1:
            for dataset_idx in [args.dataset_ID]:
        
                args.dataset_ID = dataset_idx
                args.logdir = folder+ args.datasets[args.dataset_ID]
                print('*'*10, '   Learning Shadow Models   ', '*'*10)
                args.mode_type = 'shadow'
                Train_Target_Model(args, folder='shadow')
        elif action==2:
            for dataset_idx in [args.dataset_ID]:
                print('*'*10, '  Testing Classification Accuracy of Models   ', '*'*10)
                args.dataset_ID = dataset_idx
                args.logdir = folder+ args.datasets[args.dataset_ID]
                test_models(args)

        elif action == 3:
            for dataset_idx in [args.dataset_ID]:
                print('*'*10, '  Learning adversarial examples on Target  Models ', '*'*10)
                args.dataset_ID = dataset_idx
                args.logdir = folder+ args.datasets[args.dataset_ID]
                AdversaryTwo(args)
        elif action == 4:
            for dataset_idx in [args.dataset_ID]:
                print('*'*10, '  Learning adversarial examples on Shadow  Models ', '*'*10)
                args.dataset_ID = dataset_idx
                args.logdir = folder+ args.datasets[args.dataset_ID]
                AdversaryTwo(args, folder='shadow')

        elif action == 5 :
            for dataset_idx in [args.dataset_ID]:
                args.dataset_ID = dataset_idx
                args.logdir = folder+ args.datasets[args.dataset_ID]
                print('*'*10, '  Finding minimum random noise for misclassification models ', '*'*10)
                args.max_samples=100
                AdversaryICML(args, attack=args.attack_type)
        

        elif action == 6 :
            for dataset_idx in [args.dataset_ID]:
                print('*'*10, ' The LDL (sigma) vs Classification Accuracy   ', '*'*10)
                args.dataset_ID = dataset_idx
                args.logdir = folder+ args.datasets[args.dataset_ID]
                change_Sigma(args)

        

        elif action== 7:
            print('*'*10, ' The binary attack for Location dataset ', '*'*10)

            for d in [6]:
                args.dataset_ID=d
                print(args.datasets[args.dataset_ID])
                args.logdir = folder+ args.datasets[args.dataset_ID]

                args.defense=0 
                args.mode_type = 'target'
                args.batch_size=1000
                args.epochs=50
                args.lr=0.001
                Train_Target_Model(args)
                test_models(args)

                binary_attacks(args)
                args.defense=1
                

                for sigma in [0.02,0.04,0.08,0.1]:
                    args.sigma=sigma
                    binary_attacks(args)
        elif args.action==8:
            print('*'*10, ' Learning train with L2, L1, Dropout and DF privacy defenses  ', '*'*10)
            args.dataset_ID = 0
            args.label_knowledge=True
            args.defense=False
            args.augmentation= False
            args.mode_type = 'target'
            args.logdir = 'results/CIFAR10'  
            args.Split_Size[args.dataset_ID]=[2000]
            retraining_defenses(args)
        else:
            print('*'*10, ' The action is not not defined acion value should be in range of 1 to 8  ', '*'*10)
               
    

                    
                        


if __name__ == "__main__":
    main()
