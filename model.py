import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torchvision
import torch.nn as nn
import torch
from dependency import *
from utils import get_parameter_number
import torch.nn.functional as F

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)



class FusionNet(nn.Module):

    def __init__(self, class_list):
        super(FusionNet, self).__init__()
        self.num_label = class_list[0]
        self.num_pn = class_list[1]
        self.num_str = class_list[2]
        self.num_pig = class_list[3]
        self.num_rs = class_list[4]
        self.num_dag = class_list[5]
        self.num_bwv = class_list[6]
        self.num_vs = class_list[7]
        self.dropout = nn.Dropout(0.3)
        
        self.model_clinic = torchvision.models.resnet50(pretrained=True)
        self.model_derm   = torchvision.models.resnet50(pretrained=True)

        # define the clinic model
        self.conv1_cli = self.model_clinic.conv1
        self.bn1_cli = self.model_clinic.bn1
        self.relu_cli = self.model_clinic.relu
        self.maxpool_cli = self.model_clinic.maxpool
        self.layer1_cli = self.model_clinic.layer1
        self.layer2_cli = self.model_clinic.layer2
        self.layer3_cli = self.model_clinic.layer3
        self.layer4_cli = self.model_clinic.layer4
        self.avgpool_cli = self.model_clinic.avgpool
        #self.avgpool_cli = nn.MaxPool2d(7, 7)

        self.conv1_derm = self.model_derm.conv1
        self.bn1_derm = self.model_derm.bn1
        self.relu_derm = self.model_derm.relu
        self.maxpool_derm = self.model_derm.maxpool
        self.layer1_derm = self.model_derm.layer1
        self.layer2_derm = self.model_derm.layer2
        self.layer3_derm = self.model_derm.layer3
        self.layer4_derm = self.model_derm.layer4
        self.avgpool_derm = self.model_derm.avgpool
        #self.avgpool_derm = nn.MaxPool2d(7, 7)
        # self.fc = self.model.fc
        

        self.fc_fusion_ =  nn.Sequential(
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        Swish_Module(),
        nn.Dropout(p=0.3),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        Swish_Module(),
        )
            

        self.derm_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        self.clin_mlp = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            Swish_Module(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            Swish_Module(),
        )
        

        self.fc_cli  = nn.Linear(128, self.num_label)
        self.fc_pn_cli  = nn.Linear(128, self.num_pn)
        self.fc_str_cli  = nn.Linear(128, self.num_str)
        self.fc_pig_cli  = nn.Linear(128, self.num_pig)
        self.fc_rs_cli  = nn.Linear(128, self.num_rs)
        self.fc_dag_cli  = nn.Linear(128, self.num_dag)
        self.fc_bwv_cli  = nn.Linear(128, self.num_bwv)
        self.fc_vs_cli  = nn.Linear(128, self.num_vs)


        #self.fc_derm_ = nn.Linear(2048, 512)
        self.fc_derm = nn.Linear(128, self.num_label)
        self.fc_pn_derm = nn.Linear(128, self.num_pn)
        self.fc_str_derm = nn.Linear(128, self.num_str)
        self.fc_pig_derm = nn.Linear(128, self.num_pig)
        self.fc_rs_derm = nn.Linear(128, self.num_rs)
        self.fc_dag_derm = nn.Linear(128, self.num_dag)
        self.fc_bwv_derm = nn.Linear(128, self.num_bwv)
        self.fc_vs_derm = nn.Linear(128, self.num_vs)
        # self.fc_ft = nn.

        
        self.fc_fusion = nn.Linear(128, self.num_label)
        self.fc_pn_fusion = nn.Linear(128, self.num_pn)
        self.fc_str_fusion = nn.Linear(128, self.num_str)
        self.fc_pig_fusion = nn.Linear(128, self.num_pig)
        self.fc_rs_fusion = nn.Linear(128, self.num_rs)
        self.fc_dag_fusion = nn.Linear(128, self.num_dag)
        self.fc_bwv_fusion = nn.Linear(128, self.num_bwv)
        self.fc_vs_fusion = nn.Linear(128, self.num_vs)

    def forward(self, x):
        (x_clic,x_derm) = x

        x_clic = self.conv1_cli(x_clic)
        x_clic = self.bn1_cli(x_clic)
        x_clic = self.relu_cli(x_clic)
        x_clic = self.maxpool_cli(x_clic)
        x_clic = self.layer1_cli(x_clic)
        x_clic = self.layer2_cli(x_clic)
        x_clic = self.layer3_cli(x_clic)
        x_clic = self.layer4_cli(x_clic)
        x_clic = self.avgpool_cli(x_clic)
        x_clic = x_clic.view(x_clic.size(0), -1)
        
        x_derm = self.conv1_derm(x_derm)
        x_derm = self.bn1_derm(x_derm)
        x_derm = self.relu_derm(x_derm)
        x_derm = self.maxpool_derm(x_derm)
        x_derm = self.layer1_derm(x_derm)
        x_derm = self.layer2_derm(x_derm)
        x_derm = self.layer3_derm(x_derm)
        x_derm = self.layer4_derm(x_derm)
        x_derm = self.avgpool_derm(x_derm)
        x_derm = x_derm.view(x_derm.size(0), -1)


        x_fusion = torch.add(x_clic,x_derm)
        x_fusion = self.fc_fusion_(x_fusion)

        x_clic = self.clin_mlp(x_clic)
        x_clic = self.dropout(x_clic)
        logit_clic = self.fc_cli(x_clic)
        logit_pn_clic  = self.fc_pn_cli(x_clic)
        logit_str_clic  = self.fc_str_cli(x_clic)
        logit_pig_clic  = self.fc_pig_cli(x_clic)
        logit_rs_clic  = self.fc_rs_cli(x_clic)
        logit_dag_clic  = self.fc_dag_cli(x_clic)
        logit_bwv_clic  = self.fc_bwv_cli(x_clic)
        logit_vs_clic  = self.fc_vs_cli(x_clic)

        x_derm = self.derm_mlp(x_derm)
        x_derm = self.dropout(x_derm)
        logit_derm = self.fc_derm(x_derm)
        logit_pn_derm = self.fc_pn_derm(x_derm)
        logit_str_derm = self.fc_str_derm(x_derm)
        logit_pig_derm = self.fc_pig_derm(x_derm)
        logit_rs_derm = self.fc_rs_derm(x_derm)
        logit_dag_derm = self.fc_dag_derm(x_derm)
        logit_bwv_derm = self.fc_bwv_derm(x_derm)
        logit_vs_derm = self.fc_vs_derm(x_derm)
          
        x_fusion = self.dropout(x_fusion)
        logit_fusion = self.fc_fusion(x_fusion)
        logit_pn_fusion  = self.fc_pn_fusion(x_fusion)
        logit_str_fusion  = self.fc_str_fusion(x_fusion)
        logit_pig_fusion  = self.fc_pig_fusion(x_fusion)
        logit_rs_fusion  = self.fc_rs_fusion(x_fusion)
        logit_dag_fusion  = self.fc_dag_fusion(x_fusion)
        logit_bwv_fusion  = self.fc_bwv_fusion(x_fusion)
        logit_vs_fusion  = self.fc_vs_fusion(x_fusion)

        return [(logit_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion,logit_bwv_fusion, logit_vs_fusion),
                (logit_clic , logit_pn_clic , logit_str_clic , logit_pig_clic , logit_rs_clic , logit_dag_clic , logit_bwv_clic , logit_vs_clic ),
                (logit_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm, logit_vs_derm)]



    def criterion(self, logit, truth):

        loss = nn.CrossEntropyLoss()(logit, truth)

        return loss

    def criterion1(self, logit, truth):

        loss = nn.L1Loss()(logit, truth)

        return loss


    def metric(self, logit, truth):
        # prob = F.sigmoid(logit)
        _, prediction = torch.max(logit.data, 1)

        acc = torch.sum(prediction == truth)
        return acc

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError      
             
            
            


