from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


mycfg = {
    'CNN3':  [64, 'M', 64, 'M', 128, 'M', 512, 'M'],
    'CNN4':  [64, 'M', 128, 'M', 128, 'M', 512, 'M'],
    'CNN5':  [64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'CNN6':  [64, 'M', 128, 'M', 512, 'M', 512, 'M'],
    'CNN7':  [64, 'M', 256, 'M', 512, 'M', 512, 'M'], ### targetmodel
    'CNN8':  [64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN9':  [64, 64, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN10':  [64, 128, 'M', 128, 256, 'M', 512, 'M', 512, 'M'],
    'CNN11':  [64, 128, 'M', 256, 256, 'M', 512, 'M', 512, 'M'],

}
parameters = {
    'CIFAR10':  [2, 10],
    'CIFAR100': [2, 100],
    'GTSRB':    [3, 43],
    'Face':     [5, 19],
    'TinyImageNet': [3, 200],
    'purchase': [600,100],
    'texas': [6168,100],
    'location': [446,30]
}
class CNN(nn.Module):
    def __init__(self, CNN_name, dataset, defense,sigma, dropout=0):
        super(CNN, self).__init__()
        self.dataset = dataset
        self.defense=defense
        self.layer=2
        self.features = self._make_layers(mycfg[CNN_name])
        
        self.sigma = sigma
        
        
        if dropout>0:
            self.classifier = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, parameters[self.dataset][1]) )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(True),
                nn.Linear(256, parameters[self.dataset][1]) )
        
    def forward(self, x):
        
        pred=[]
        if self.defense:
            
            for i in range(x.shape[0]):
                noise=torch.normal(mean=0, std=self.sigma, size=(200,x.shape[1],x.shape[2],x.shape[3])).cuda()
                # print(self.sigma)
                x1=x[i]+noise

                
                if self.layer == 1 :
                    p = torch.mean(self.features(x1),0,True)
                    p = p.view(p.size(0), -1)
                    p = self.classifier(p)
                if self.layer == 2 :
                    p = self.features(x1)
                    p = p.view(p.size(0), -1)
                    p = torch.mean( self.classifier(p),axis=0)

                pred.append(p)
            out=torch.stack(pred, dim=0)
            
            out=torch.squeeze(out)
            
        else: 

            out=self.features(x)
            out = out.view(out.size(0), -1)
            out = self.classifier(out)

       
        return out

        def predict(self,x):
            return self.forward(x)



    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           #nn.BatchNorm2d(x, track_running_stats=True),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=parameters[self.dataset][0], stride=parameters[self.dataset][0])]
        return nn.Sequential(*layers)

class fc_connect(nn.Module):
    def __init__(self, dataset, defense,sigma, dropout=False):
        super(fc_connect, self).__init__()
        self.dataset = dataset
        self.defense=defense
        self.sigma = sigma
        
        self.classifier = nn.Sequential(
            nn.Linear(parameters[self.dataset][0], 128),
            nn.Tanh(),
            # nn.Linear(128, 128),
            # nn.Tanh(),
            # nn.Linear(128, 128),
            # nn.Tanh(),
            nn.Linear(128, parameters[self.dataset][1]) )
    def forward(self,x):
        pred=[]
        if self.defense:
            
            for i in range(x.shape[0]):
               
                x_temp=x[i:i+1].detach().cpu().numpy().astype(np.bool)
                noise = np.random.binomial(1, self.sigma, [100, x_temp.shape[-1]])
                x_sampled = np.tile(np.copy(x_temp), (100, 1))
                x_noisy = np.invert(x_temp, out=x_sampled,where=noise.astype(np.bool)).astype(np.int32)
                
                p = self.classifier(torch.from_numpy(x_noisy.astype(np.float32)).cuda())
                p = torch.mean( p,axis=0)

                pred.append(p)

            out=torch.stack(pred, dim=0)
            
        else: 

            out = self.classifier(x)
        return out

class MemGuard(nn.Module):
    def __init__(self):
        super(MemGuard, self).__init__()

    def forward(self, logits):
        scores = F.softmax(logits, dim=1)#.cpu().numpy()
        n_classes = scores.shape[1]
        epsilon = 1e-3
        on_score = (1. / n_classes) + epsilon
        off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
        predicted_labels = scores.max(1)[1]
        defended_scores = torch.ones_like(scores) * off_score
        defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
        return defended_scores
     
