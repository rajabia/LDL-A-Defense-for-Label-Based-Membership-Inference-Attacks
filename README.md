# LDL-A-Defense-for-Label-Based-Membership-Inference-Attacks
These codes are for [LDL: A Defense for Label-Based Membership Inference Attacks](https://arxiv.org/pdf/2212.01688). LDL provides a defense against label-based membership inference attacks by creating a label-invariance sphere around an input sample. 

To run the code


```ruby

python mai.py --dataset_ID 0  --action 0

--action 0-8 please read the main file to actions function (action 0 and 1 trainhe taget model and substitude model without any defense respectively for a given dataset)
action 3 tests the target and shadow models again label-based membership inference attack  which uses HopSkipJum. To turn the LDL defense set the defense to and and use your sigma values (--defense 1 --sigma 0.02)
```



```ruby

@article{rajabi2023LDL
  title={LDL: A Defense for Label-Based Membership Inference Attacks},
  author={Rajabi, Arezoo and Sahabandu, Dinuka amd Niu, Luyao and  Ramasubramanian, Bhaskar and and Poovendran, Radha},
  journal={18th ACM ASIA Conference on Computer and Communications Security (ACM ASIACCS)},
  year={2023}
}
```
