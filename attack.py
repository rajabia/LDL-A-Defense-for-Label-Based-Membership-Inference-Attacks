import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
from runx.logx import logx

from foolbox.distances import l0, l1, l2, linf
import math
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score 

from art.attacks.evasion import HopSkipJump
from art.estimators.classification import PyTorchClassifier
from art.utils import compute_success

# import QEBA
# from QEBA.criteria import TargetClass, Misclassification
# from QEBA.pre_process.attack_setting import load_pgen

from utils import create_rotates, apply_augment, train_attack_model, get_max_accuracy
import os


from utils import train_attack_model


def prediction(x):
    x_list = x[0].tolist()
    x_sort = sorted(x_list)
    max_index = x_list.index(x_sort[-1])

    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum

    return softmax, max_index#, sec_index

def AdversaryOne_Feature(args, shadowmodel, data_loader, cluster, Statistic_Data):
    Loss = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.cuda()
            output = shadowmodel(data)
            Loss.append(F.cross_entropy(output, target.cuda()).item())
    Loss = np.asarray(Loss)
    half = int(len(Loss)/2)
    member = Loss[:half]
    non_member = Loss[half:]        
    for loss in member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Member'})
    for loss in non_member:
        Statistic_Data.append({'DataSize':float(cluster), 'Loss':loss,  'Status':'Non-member'})
    return Statistic_Data


def AdversaryOne_evaluation(args, targetmodel, shadowmodel, data_loader, cluster, AUC_Loss, AUC_Entropy, AUC_Maximum):
    Loss = []
    Entropy = []
    Maximum = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.cuda(), target.cuda()
            Toutput = targetmodel(data)
            Tlabel = Toutput.max(1)[1]

            Soutput = shadowmodel(data)
            if Tlabel != target:
               
                Loss.append(100)
            else:
                Loss.append(F.cross_entropy(Soutput, target).item())
            
            prob = F.softmax(Soutput, dim=1) 

            Maximum.append(torch.max(prob).item())
            entropy = -1 * torch.sum(torch.mul(prob, torch.log(prob)))
            if str(entropy.item()) == 'nan':
                Entropy.append(1e-100)
            else:
                Entropy.append(entropy.item())
 
    mem_groundtruth = np.ones(int(len(data_loader.dataset)/2))
    non_groundtruth = np.zeros(int(len(data_loader.dataset)/2))
    groundtruth = np.concatenate((mem_groundtruth, non_groundtruth))

    predictions_Loss = np.asarray(Loss)
    predictions_Entropy = np.asarray(Entropy)
    predictions_Maximum = np.asarray(Maximum)
    
    fpr, tpr, _ = roc_curve(groundtruth, predictions_Loss, pos_label=0, drop_intermediate=False)
    AUC_Loss.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Entropy, pos_label=0, drop_intermediate=False)
    AUC_Entropy.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})

    fpr, tpr, _ = roc_curve(groundtruth, predictions_Maximum, pos_label=1, drop_intermediate=False)
    AUC_Maximum.append({'DataSize':float(cluster), 'AUC':round(auc(fpr, tpr), 4)})
    return AUC_Loss, AUC_Entropy, AUC_Maximum

def AdversaryTwo_HopSkipJump(args, targetmodel, data_loader, cluster, AUC_Dist, Distance, maxitr=50, max_eval=10000):
    input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128)]
    nb_classes = [10, 100, 43, 19]
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                # clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=input_shape[args.dataset_ID],
                nb_classes=nb_classes[args.dataset_ID],
            )
    L0_dist, L1_dist, L2_dist, Linf_dist = [], [], [], []
    Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, max_iter=maxitr, max_eval=max_eval)

    mid = int(len(data_loader)/2)
    member_groundtruth, non_member_groundtruth = [], []
    success_list=[]
    
    for idx, (data, target) in enumerate(data_loader): 
        success=0
        
        data = np.array(data) 
        logit = ARTclassifier.predict(data)
        pred=np.argmax(logit)
        
        
        if not args.label_knowledge:
            target= torch.tensor(pred)
        
        if pred != target.item():
            
            success = 1
            data_adv = data
            
        else:
            data_adv = Attack.generate(x=data) 
            data_adv = np.array(data_adv) 
            
            success = compute_success(ARTclassifier, data, [target.item()], data_adv)
            

        if success == 1:
            
            logx.msg('-------------Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            L0_dist.append(l0(data, data_adv))
            L1_dist.append(l1(data, data_adv))
            L2_dist.append(l2(data, data_adv))
            Linf_dist.append(linf(data, data_adv))

            
        else:
            logx.msg('----Not a Succes--- Training DataSize: {} current img index:{}---------------'.format(cluster, idx))
            L0_dist.append(100000)
            L1_dist.append(100000)
            L2_dist.append(100000)
            Linf_dist.append(100000)

        success_list.append(success)
        if idx < mid:
            member_groundtruth.append(1)
        else:
            non_member_groundtruth.append(0)

        
        
    member_groundtruth = np.array(member_groundtruth)
    non_member_groundtruth = np.array(non_member_groundtruth)

    groundtruth = np.concatenate((member_groundtruth, non_member_groundtruth))
    L0_dist = np.array(L0_dist)
    L1_dist = np.array(L1_dist)
    L2_dist = np.array(L2_dist)
    Linf_dist = np.array(Linf_dist)

    fpr, tpr, _ = roc_curve(groundtruth, L0_dist, pos_label=1, drop_intermediate=False)
    L0_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L1_dist, pos_label=1, drop_intermediate=False)
    L1_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, L2_dist, pos_label=1, drop_intermediate=False)
    L2_auc = round(auc(fpr, tpr), 4)
    fpr, tpr, _ = roc_curve(groundtruth, Linf_dist, pos_label=1, drop_intermediate=False)
    Linf_auc = round(auc(fpr, tpr), 4)

    ### AUC based on distance
    auc_score = {'DataSize':float(cluster), 'L0_auc':L0_auc, 'L1_auc':L1_auc, 'L2_auc':L2_auc, 'Linf_auc':Linf_auc}
    AUC_Dist.append(auc_score)

    ### Distance of L0, L1, L2, Linf
    middle= int(len(L0_dist)/2)
    for idx, (l0_dist, l1_dist, l2_dist, linf_dist) in enumerate(zip(L0_dist, L1_dist, L2_dist, Linf_dist)):   
        if idx < middle:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Member'}
        else:
            data = {'DataSize':float(cluster), 'L0Distance':l0_dist, 'L1Distance':l1_dist, 'L2Distance':l2_dist, 'LinfDistance':linf_dist, 'Status':'Non-member'}
        Distance.append(data)
    print(AUC_Dist)
    print('Success Rate of Learning Adversarial Examples was:   ', sum(success_list)/len(success_list) )
    
    return AUC_Dist, Distance

def HopSkipJump_for_icml(args, targetmodel, data_loader, maxitr=50, max_eval=10000):
    input_shape = [(3, 32, 32), (3, 32, 32), (3, 64, 64), (3, 128, 128)]
    nb_classes = [10, 100, 43, 19]
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                # clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=input_shape[args.dataset_ID],
                nb_classes=nb_classes[args.dataset_ID],
            )
    L2_dist = []
    Attack = HopSkipJump(classifier=ARTclassifier, targeted =False, max_iter=maxitr, max_eval=max_eval)

    
    success_list=[]
    for idx in range(len(data_loader[0])): 
        data, target = data_loader[0][idx], data_loader[1][idx]
        success=0

        data = np.array(data) 
        logit = ARTclassifier.predict(data)
        
        pred=np.argmax(logit)
        if not args.label_knowledge:
            target= torch.tensor(pred)
        if pred != target.item():
            
            success = 1
            data_adv = data
            
        else:
            data_adv = Attack.generate(x=data) 
            data_adv = np.array(data_adv) 
            
            success = compute_success(ARTclassifier, data, [target.item()], data_adv)
            

        if success == 1:
            
            logx.msg('------------- current img index:{}---------------'.format(idx))
            
            L2_dist.append(l2(data, data_adv))

            
        else:
            logx.msg('----Not a Succes---  current img index:{}---------------'.format(idx))
            
            L2_dist.append(100000)
    

        success_list.append(success)
        
        
    
    
    L2_dist = np.asarray(L2_dist).squeeze()

    
    return  L2_dist

def distance_augmentation_attack(args,model, train_set, test_set, attack_type='d', augment_kwarg=1, batch=100, input_dim=[None, 32, 32, 3], n_classes=10):
    """Combined MI attack using the distanes for each augmentation.
    Args:
    model: model to approximate distances on (attack).
    train_set: the training set for the model
    test_set: the test set for the model
    max_samples: max number of samples to attack
    attack_type: either 'd' or 'r' for translation and rotation attacks, respectively.
    augment_kwarg: the kwarg for each augmentation. If rotations, augment_kwarg defines the max rotation, with n=2r+1
    rotated images being used. If translations, then 4n+1 translations will be used at a max displacement of
    augment_kwarg
    batch: batch size for querying model in the attack.
    Returns: 2D array where rows correspond to samples and columns correspond to the distance to boundary in an untargeted
    attack for that rotated/translated sample.
    """
    if attack_type == 'r':
        augments = create_rotates(augment_kwarg)
    elif attack_type == 'd':
        augments = create_translates(augment_kwarg)
    else:
        raise ValueError(f"attack type_: {attack_type} is not valid.")
    max_samples=args.max_samples
    m = np.concatenate([np.ones(max_samples), np.zeros(max_samples)], axis=0)
    

    attack_in = np.zeros((max_samples, len(augments)))
    attack_out = np.zeros((max_samples, len(augments)))
    print(attack_in.shape,max_samples, augments)

    for i, augment in enumerate(augments):
        train_augment = apply_augment(train_set, augment, attack_type)
        test_augment = apply_augment(test_set, augment, attack_type)
        print(train_augment[0].shape, len(train_augment))
        # (args, targetmodel, data_loader, maxitr=50, max_eval=10000):
        l2_train = HopSkipJump_for_icml(args, model, train_augment, maxitr=150, max_eval=10000)
        print('@'*10+'L2 distance for augments '+str(i)+ '   ', l2_train)
        attack_in[:, i] = l2_train
        l2_test = HopSkipJump_for_icml(args, model, test_augment, maxitr=150, max_eval=10000)
        attack_out[:, i] = l2_test
        
        
        
        
    
    attack_set = (np.concatenate([attack_in, attack_out], 0),m.astype(np.int16))
    return attack_set


def continuous_rand_robust(args,targetmodel, dataloader, max_samples=100, noise_samples=2500, stddev=0.025, 
                            num=[1, 5, 10, 20, 50,  100,  200, 500, 700, 1000]):
    # num=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 350, 500, 750, 1000]
    """Calculate robustness to random noise for Adv-x MI attack on continuous-featureed datasets (+ UCI adult).
    :param model: model to approximate distances on (attack).
    :param ds: tf dataset should be either the training set or the test set.
    :param max_samples: maximum number of samples to take from the ds
    :param noise_samples: number of noised samples to take for each sample in the ds.
    :param stddev: the standard deviation to use for Gaussian noise (only for Adult, which has some continuous features)
    :param input_dim: dimension of inputs for the dataset.
    :param num: subnumber of samples to evaluate. max number is noise_samples
    :return: a list of lists. each sublist of the accuracy on up to $num noise_samples.
    """
    

    num_samples = 0
    robust_accs = [[] for _ in num]
    all_correct = []
    input_shape = [[3, 32, 32], [3, 32, 32], [3, 64, 64], [3, 128, 128]]
    nb_classes = [10, 100, 43, 19]
    ARTclassifier = PyTorchClassifier(
                model=targetmodel,
                # clip_values=(0, 1),
                loss=F.cross_entropy,
                input_shape=input_shape[args.dataset_ID],
                nb_classes=nb_classes[args.dataset_ID],
            )

    dims=input_shape[args.dataset_ID]
    
  
    
    for idx in range(len(dataloader[0])): 
        data, target = dataloader[0][idx], dataloader[1][idx]
        
        # print('dasetID: '+str(args.dataset_ID)+ '  Sigma: '+str(stddev)+'  Sample  '+str(idx)+'  From '+str(len(dataloader[0])))
        data = np.array(data) 
        logit = ARTclassifier.predict(data)
    
        pred=np.argmax(logit)
        if not args.label_knowledge:
            target= torch.tensor(pred)

        correct = pred == target.item()
        all_correct.append(correct)
        
        if correct:
            noise = stddev * np.random.randn(noise_samples,dims[0],dims[1],dims[2])
            x_noisy = np.clip(data + noise, 0, 1).astype(np.float32)
            

            logit = ARTclassifier.predict(x_noisy)
            pred=np.argmax(logit,axis=1)
            
            for idx, n in enumerate(num):
                if n == 0:
                    robust_accs[idx].append(1)
                else:
                    robust_accs[idx].append(np.mean(pred[:n] == target))
        else:
            for idx in range(len(num)):
                robust_accs[idx].append(0)

            
    if args.label_knowledge:
        prifix= ''
    else:
        prifix= 'WeakAdv_'
        
    if args.defense:
        pathadd=args.logdir + '/'+str(args.sigma)+'/'+prifix+'attack_g.npy' 
        
    else:
        pathadd=args.logdir + '/'+str(args.sigma)+'/NoDefense_'+prifix+'attack_g.npy'
    
    np.save(pathadd, robust_accs)
    return robust_accs


def binary_rand_robust(args,model, dataloader, p,  noise_samples=1000):
    
    
    robust_accs = []
    
        
    for x,y in dataloader:
        pred=model(x.cuda())
        correct=pred.max(1)[1].cpu()==y
        
        if correct:
            temp=x.detach().cpu().numpy()
            noise = np.random.binomial(1, p, [noise_samples, temp.shape[-1]])
            x_sampled = np.tile(np.copy(temp), (noise_samples, 1))
            x_noisy = np.invert(temp.astype(np.bool), out=x_sampled,where=noise.astype(np.bool)).astype(np.int32)
            
            preds=model(torch.from_numpy(x_noisy.astype(np.float32)).cuda()).max(1)[1]
            preds=preds.cpu().numpy()

                
            robust_accs.append(np.mean(preds == y.item()))

        else:
            
            robust_accs.append(0)

        

        
    if not os.path.exists(args.logdir + '/'+str(args.sigma)):
        os.mkdir(args.logdir + '/'+str(args.sigma))
    if args.defense:
        pathadd=args.logdir + '/'+str(args.sigma)+'/'+'attack_b.npy' 
        
    else:
        pathadd=args.logdir + '/'+str(args.sigma)+'/NoDefense_'+'attack_b.npy'
    
    np.save(pathadd, robust_accs)

    return robust_accs

def torch_to_numpy(dataloader, nsamples):
    all_X = []
    all_y = []
    for X, y in dataloader:
        all_X.append(X.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())
        if len(all_X)== nsamples:
            break
    
    return (np.array(all_X),np.array(all_y))


def icml_attack(target_model,train_loader_target, test_loader_target, source_model, train_loader_shadow, test_loader_shadow,AUC_Dist,cluster, attack,aug_kwarg,args):
    # if attack == 'n':
    #   # just look at confidence in predicted label
    #     conf_source = np.max(source_features, axis=-1)
    #     conf_target = np.max(target_features, axis=-1)
    #     print("threshold on predicted label:")
    #     acc1, prec1, _, _ = get_threshold(source_m, conf_source, target_m, conf_target)

    #     # look at confidence in true label
    #     conf_source = source_features[range(len(source_features)), source_labels]
    #     conf_target = target_features[range(len(target_features)), target_labels]
    #     print("threshold on true label:")
    #     acc2, prec2, _, _ = get_threshold(source_m, conf_source, target_m, conf_target)
    input_shape = [[3, 32, 32], [3, 32, 32], [3, 64, 64], [3, 128, 128]]
    input_dim=input_shape[args.dataset_ID]
    n_classes = args.num_classes[args.dataset_ID]
    
    args.attack_batch_size=1
    max_samples= args.max_samples

    target_train_set = torch_to_numpy(train_loader_target, max_samples)
    target_test_set = torch_to_numpy(test_loader_target, max_samples)

    source_train_set = torch_to_numpy(train_loader_shadow, max_samples)
    source_test_set = torch_to_numpy(test_loader_shadow, max_samples)

    if attack == 'd' or attack == 'r':
        
        
        aug_kwarg = args.d if attack == 'd' else args.r
        attack_test_set = distance_augmentation_attack(args,target_model, target_train_set, target_test_set,attack,
                                                        aug_kwarg, args.attack_batch_size,
                                                        input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],n_classes=n_classes)
        attack_train_set =distance_augmentation_attack(args,source_model, source_train_set, 
                                                        source_test_set,
                                                        attack,aug_kwarg, args.attack_batch_size,
                                                        input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],n_classes=n_classes)

        train_acc, test_acc = train_attack_model(args,attack_train_set, attack_test_set, attack)
        

        auc_score = {'DataSize':float(cluster), 'Shadow Model Accuracy':train_acc, 'Target Model Accuracy':test_acc}
        AUC_Dist.append(auc_score)
        if args.label_knowledge:
            prifix= ''
        else:
            prifix= 'WeakAdv_'
            
        if args.defense:
            
            pathadd=args.logdir + '/adversaryTwo/data_'+prifix+attack + '.npy'
            
        else:
            pathadd=args.logdir + '/adversaryTwo/NoDefensedata_'+prifix+attack+ '.npy'

        np.save(pathadd,{'training_x':attack_train_set[0], 'training_y':attack_train_set[1], 'test_x':attack_test_set[0],'test_y':attack_test_set[1]})
        return AUC_Dist
    elif attack== 'g':

        if args.label_knowledge:
            prifix= ''
        else:
            prifix= 'WeakAdv_'
        args.noise_samples = 1000
        # target_m = np.concatenate([np.ones(len(target_train_ds[0])),np.zeros(len(target_test_ds[0]))], axis=0)
        print(args.logdir +'/'+str(args.sigma)+'./g_'+str(args.dataset_ID)+'_cl_'+str(cluster)+'.npy')
        dict_res={'sigma':[0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  0.08,  0.1], 'AUC':[]}
        for sigma in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,  0.08,  0.1]:
            print(f"threshold on noise robustness, sigma: {sigma}")
            
            # noise_source_in = continuous_rand_robust(source_model, source_train_ds, stddev=sigma,  input_dim=[None, input_dim], noise_samples=args.noise_samples)
            # noise_source_out = continuous_rand_robust(source_model, source_test_ds, stddev=sigma, input_dim=[None, input_dim], noise_samples=args.noise_samples)
            
            # args,targetmodel, dataloader, max_samples=100, noise_samples=2500, stddev=0.025, 
            noise_target_in = continuous_rand_robust(args,target_model, target_train_set, stddev=sigma, noise_samples=args.noise_samples)
            noise_target_out = continuous_rand_robust(args,target_model, target_test_set, stddev=sigma,  noise_samples=args.noise_samples)

            
            
            if args.defense:
            
                pathadd=args.logdir + '/'+str(args.sigma)+'/data_'+prifix+attack + '_sigma_'+str(sigma)+'_'+str(cluster)+'_new.npy'
            
            else:
                pathadd=args.logdir + '/'+str(args.sigma)+'/NoDefensedata_'+prifix+attack+ '_sigma_'+str(sigma)+'_'+str(cluster)+'_new.npy'

            np.save(pathadd,{'noise_target_in':noise_target_in, 'noise_target_out':noise_target_out})
            res1,res2=[],[]
            for i in range(len(noise_target_in)):
                # noise_source = np.concatenate([noise_source_in[i], noise_source_out[i]], axis=0)
                
                noise_target_in=np.array(noise_target_in)
                noise_target_out= np.array(noise_target_out)
                noise_target = np.concatenate([noise_target_in[i], noise_target_out[i]], axis=0)
                y_true= np.array([1]*len(noise_target_in[i])+[0]*len(noise_target_out[i]))
                #max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold
                # max_accuracy, max_accuracy_threshold, max_precision, max_precision_threshold=get_max_accuracy(y_true, noise_target, thresholds=None)
                # res1.append(max_accuracy)
                auc=roc_auc_score(y_true, noise_target)
                res2.append(auc)
                

            print(sigma, res2)
            # dict_res['max_accuracy'].append(res1)
            dict_res['AUC'].append(res2)
        if args.defense:
            np.save(args.logdir +'/'+str(args.sigma)+'/g_'+str(args.dataset_ID)+'_cl_'+str(cluster)+'_new.npy',dict_res)  
        else:
            np.save(args.logdir +'/'+str(args.sigma)+'/NoDefense_g_'+str(args.dataset_ID)+'_cl_'+str(cluster)+'_new.npy',dict_res)
        
        AUC_Dist.append(dict_res)

        return AUC_Dist


