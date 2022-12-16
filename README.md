# LDL-A-Defense-for-Label-Based-Membership-Inference-Attacks
These codes are for [LDL: A Defense for Label-Based Membership Inference Attacks](https://arxiv.org/pdf/2212.01688). LDL provides a defense against label-based membership inference attacks by creating a label-invariance sphere around an input sample. 

To run the code


```ruby
python mai.py --dataset_ID 0  --action 0
```

We defined 9 different actions (0-8). 
Actions 0 and 1 train the taget model and substitude model without any defense respectively for a specified dataset (ID of each dataset CIFAR10: 0 , CIFAR100: 1, GTSRB: 2 and Fac3: 3). The training hyperparameter such as number of epochs, batchsize learning rate, etc. can be given.
Action 3 tests the target and shadow models against the label-based membership inference attack which uses HopSkipJum. To turn the LDL defense for avaluating against MIA attacks, set the defense to 1 and and use your desired sigma values (--defense 1 --sigma 0.02)
Please read the main file to find the actions' function.

We provided the code for sampling from LFW dataset for Face dataset in lfw.py 

The adversarialRobustenss.py provides the codes for evaluating the the LDL defense with different sigma againt adversarial learning method of fast gradient sign.



```ruby

@article{rajabi2023LDL
  title={LDL: A Defense for Label-Based Membership Inference Attacks},
  author={Rajabi, Arezoo and Sahabandu, Dinuka amd Niu, Luyao and  Ramasubramanian, Bhaskar and and Poovendran, Radha},
  journal={18th ACM ASIA Conference on Computer and Communications Security (ACM ASIACCS)},
  year={2023}
}
```
