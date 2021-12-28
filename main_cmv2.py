import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from utils import Logger, adjust_learning_rate, CraateLogger,create_cosine_learing_schdule,encode_test_label,set_seed
from model import FusionNet
from dependency import *
from torch import optim
from torchcontrib.optim import SWA
from dataloader import generate_dataloader
import torch
import numpy as np
import torch.nn.functional as F


def train(net,train_dataloader,model_name):

    net.set_mode('train')
    train_loss = 0
    train_dia_acc = 0  
    train_sps_acc = 0
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(train_dataloader):
        opt.zero_grad()
        
        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
        meta_data    = meta_data.cuda()
        
        # Diagostic label
        diagnosis_label = label[0].long().cuda()
        # Seven-Point Checklikst labels
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()


        [(logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm, 
          logit_vs_derm), 
         (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic,
          logit_vs_clic),  
         (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion, 
          logit_vs_fusion) ] = net((clinic_image,derm_image))
        
        
        loss_fusion = torch.true_divide(
            net.criterion(logit_diagnosis_fusion, diagnosis_label)
            + net.criterion(logit_pn_fusion, pn_label)
            + net.criterion(logit_str_fusion, str_label)
            + net.criterion(logit_pig_fusion, pig_label)
            + net.criterion(logit_rs_fusion, rs_label)
            + net.criterion(logit_dag_fusion, dag_label)
            + net.criterion(logit_bwv_fusion, bwv_label)
            + net.criterion(logit_vs_fusion, vs_label), 8)

        loss_clic = torch.true_divide(
            net.criterion(logit_diagnosis_clic, diagnosis_label)
            + net.criterion(logit_pn_clic, pn_label)
            + net.criterion(logit_str_clic, str_label)
            + net.criterion(logit_pig_clic, pig_label)
            + net.criterion(logit_rs_clic, rs_label)
            + net.criterion(logit_dag_clic, dag_label)
            + net.criterion(logit_bwv_clic, bwv_label)
            + net.criterion(logit_vs_clic, vs_label), 8)

        loss_derm = torch.true_divide(
            net.criterion(logit_diagnosis_derm, diagnosis_label)
            + net.criterion(logit_pn_derm, pn_label)
            + net.criterion(logit_str_derm, str_label)
            + net.criterion(logit_pig_derm, pig_label)
            + net.criterion(logit_rs_derm, rs_label)
            + net.criterion(logit_dag_derm, dag_label)
            + net.criterion(logit_bwv_derm, bwv_label)
            + net.criterion(logit_vs_derm, vs_label), 8)


        loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33 

        dia_acc_fusion = torch.true_divide(net.metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
        dia_acc_clic = torch.true_divide(net.metric(logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
        dia_acc_derm = torch.true_divide(net.metric(logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))

        dia_acc = torch.true_divide(dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)

        sps_acc_fusion = torch.true_divide(net.metric(logit_pn_fusion, pn_label)
                                     + net.metric(logit_str_fusion, str_label)
                                     + net.metric(logit_pig_fusion, pig_label)
                                     + net.metric(logit_rs_fusion, rs_label)
                                     + net.metric(logit_dag_fusion, dag_label)
                                     + net.metric(logit_bwv_fusion, bwv_label)
                                     + net.metric(logit_vs_fusion, vs_label), 7 * clinic_image.size(0))
        sps_acc_clic = torch.true_divide(net.metric(logit_pn_clic, pn_label)
                                     + net.metric(logit_str_clic, str_label)
                                     + net.metric(logit_pig_clic, pig_label)
                                     + net.metric(logit_rs_clic, rs_label)
                                     + net.metric(logit_dag_clic, dag_label)
                                     + net.metric(logit_bwv_clic, bwv_label)
                                     + net.metric(logit_vs_clic, vs_label), 7 * clinic_image.size(0))
        sps_acc_derm = torch.true_divide(net.metric(logit_pn_derm, pn_label)
                                     + net.metric(logit_str_derm, str_label)
                                     + net.metric(logit_pig_derm, pig_label)
                                     + net.metric(logit_rs_derm, rs_label)
                                     + net.metric(logit_dag_derm, dag_label)
                                     + net.metric(logit_bwv_derm, bwv_label)
                                     + net.metric(logit_vs_derm, vs_label), 7 * clinic_image.size(0))

        sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)


        loss.backward()
        opt.step()

        train_loss += loss.item()
        train_dia_acc += dia_acc.item()
        train_sps_acc += sps_acc.item()

    train_loss = train_loss / (index + 1) # Because the index start with the value 0f zero
    train_dia_acc = train_dia_acc / (index + 1)
    train_sps_acc = train_sps_acc / (index + 1)

    return train_loss,train_dia_acc,train_sps_acc

def validation(net,val_dataloader,model_name):
    net.set_mode('valid')
    val_loss = 0
    val_dia_acc = 0
    vaL_sps_acc = 0
    for index, (clinic_image, derm_image, meta_data, label) in enumerate(val_dataloader):

        clinic_image = clinic_image.cuda()
        derm_image   = derm_image.cuda()
        meta_data    = meta_data.cuda()

        diagnosis_label = label[0].long().cuda()
        pn_label = label[1].long().cuda()
        str_label = label[2].long().cuda()
        pig_label = label[3].long().cuda()
        rs_label = label[4].long().cuda()
        dag_label = label[5].long().cuda()
        bwv_label = label[6].long().cuda()
        vs_label = label[7].long().cuda()

        with torch.no_grad():


          [(logit_diagnosis_derm, logit_pn_derm, logit_str_derm, logit_pig_derm, logit_rs_derm, logit_dag_derm, logit_bwv_derm, 
              logit_vs_derm), 
              (logit_diagnosis_clic, logit_pn_clic, logit_str_clic, logit_pig_clic, logit_rs_clic, logit_dag_clic, logit_bwv_clic,
              logit_vs_clic),  
              (logit_diagnosis_fusion, logit_pn_fusion, logit_str_fusion, logit_pig_fusion, logit_rs_fusion, logit_dag_fusion, logit_bwv_fusion, 
              logit_vs_fusion)] = net((clinic_image,derm_image))
  
          loss_fusion = torch.true_divide(
              net.criterion(logit_diagnosis_fusion, diagnosis_label)
              + net.criterion(logit_pn_fusion, pn_label)
              + net.criterion(logit_str_fusion, str_label)
              + net.criterion(logit_pig_fusion, pig_label)
              + net.criterion(logit_rs_fusion, rs_label)
              + net.criterion(logit_dag_fusion, dag_label)
              + net.criterion(logit_bwv_fusion, bwv_label)
              + net.criterion(logit_vs_fusion, vs_label), 8)
  
          loss_clic = torch.true_divide(
              net.criterion(logit_diagnosis_clic, diagnosis_label)
              + net.criterion(logit_pn_clic, pn_label)
              + net.criterion(logit_str_clic, str_label)
              + net.criterion(logit_pig_clic, pig_label)
              + net.criterion(logit_rs_clic, rs_label)
              + net.criterion(logit_dag_clic, dag_label)
              + net.criterion(logit_bwv_clic, bwv_label)
              + net.criterion(logit_vs_clic, vs_label), 8)
  
          loss_derm = torch.true_divide(
              net.criterion(logit_diagnosis_derm, diagnosis_label)
              + net.criterion(logit_pn_derm, pn_label)
              + net.criterion(logit_str_derm, str_label)
              + net.criterion(logit_pig_derm, pig_label)
              + net.criterion(logit_rs_derm, rs_label)
              + net.criterion(logit_dag_derm, dag_label)
              + net.criterion(logit_bwv_derm, bwv_label)
              + net.criterion(logit_vs_derm, vs_label), 8)
  
  
          loss = loss_fusion*0.33 + loss_clic*0.33 + loss_derm*0.33 
  
          dia_acc_fusion = torch.true_divide(net.metric(logit_diagnosis_fusion, diagnosis_label), clinic_image.size(0))
          dia_acc_clic = torch.true_divide(net.metric(logit_diagnosis_clic, diagnosis_label), clinic_image.size(0))
          dia_acc_derm = torch.true_divide(net.metric(logit_diagnosis_derm, diagnosis_label), clinic_image.size(0))
  
          dia_acc = torch.true_divide(dia_acc_fusion + dia_acc_clic + dia_acc_derm, 3)
  
          sps_acc_fusion = torch.true_divide(net.metric(logit_pn_fusion, pn_label)
                                       + net.metric(logit_str_fusion, str_label)
                                       + net.metric(logit_pig_fusion, pig_label)
                                       + net.metric(logit_rs_fusion, rs_label)
                                       + net.metric(logit_dag_fusion, dag_label)
                                       + net.metric(logit_bwv_fusion, bwv_label)
                                       + net.metric(logit_vs_fusion, vs_label), 7 * clinic_image.size(0))

          sps_acc_clic = torch.true_divide(net.metric(logit_pn_clic, pn_label)
                                       + net.metric(logit_str_clic, str_label)
                                       + net.metric(logit_pig_clic, pig_label)
                                       + net.metric(logit_rs_clic, rs_label)
                                       + net.metric(logit_dag_clic, dag_label)
                                       + net.metric(logit_bwv_clic, bwv_label)
                                       + net.metric(logit_vs_clic, vs_label), 7 * clinic_image.size(0))
                                       
          sps_acc_derm = torch.true_divide(net.metric(logit_pn_derm, pn_label)
                                       + net.metric(logit_str_derm, str_label)
                                       + net.metric(logit_pig_derm, pig_label)
                                       + net.metric(logit_rs_derm, rs_label)
                                       + net.metric(logit_dag_derm, dag_label)
                                       + net.metric(logit_bwv_derm, bwv_label)
                                       + net.metric(logit_vs_derm, vs_label), 7 * clinic_image.size(0))
  
          sps_acc = torch.true_divide(sps_acc_fusion + sps_acc_clic + sps_acc_derm, 3)



        val_loss += loss.item()
        val_dia_acc += dia_acc.item()
        vaL_sps_acc += sps_acc.item()

    val_loss = val_loss / (index + 1)
    val_dia_acc = val_dia_acc / (index + 1)
    vaL_sps_acc = vaL_sps_acc / (index + 1)

    return val_loss,val_dia_acc,vaL_sps_acc


def run_train(model_name,mode,i):
    log.write('** start training here! **\n')
    #best_acc = 0
    es = 0
    patience = 50
    best_mean_acc = 0 
    best_loss = 300
    
    for epoch in range(epochs):
        swa_lr = cosine_learning_schule[epoch]
        adjust_learning_rate(opt, swa_lr)

        # train_mode
        train_loss,train_dia_acc,train_sps_acc=train(net, train_dataloader,model_name)
        log.write('Round: {}, epoch: {}, Train Loss: {:.4f}, Train Dia Acc: {:.4f}, Train SPS Acc: {:.4f}\n'.format(i, epoch, train_loss,
                                                                                                         train_dia_acc,
                                                                                                         train_sps_acc))

        # validation mode
        val_loss,val_dia_acc,val_sps_acc = validation(net, val_dataloader,model_name)
        
        val_acc = (val_dia_acc + val_sps_acc) / 2
        val_mean_acc = (val_dia_acc*1 + val_sps_acc*7)/8
        
        log.write('Round: {}, epoch: {}, Valid Loss: {:.4f}, Valid Dia Acc: {:.4f}, Valid SPS Acc: {:.4f}\n'.format(i, epoch, val_loss,
                                                                                                         val_dia_acc,
                                                                                                         val_sps_acc))

     
        if val_mean_acc > best_mean_acc:
            es = 0
            best_mean_acc = val_mean_acc
            torch.save(net.state_dict(), out_dir + '/checkpoint/{}_model.pth'.format('best_mean_acc'))
            log.write('Current Best Mean Acc is {}'.format(best_mean_acc))
        #  else:
        #      es += 1
        #      print("Counter {} of {}".format(es,patience))
          
        #      if es > patience:
        #          print("Early stopping with best_mean_acc: {:.4f}".format(best_mean_acc), "and val_mean_acc for this epoch: {:.4f}".format(val_mean_acc))
        #          break
  
  
        if epoch > (epochs - swa_epoch) and epoch % 1 == 0:
            opt.update_swa()
            log.write('SWA Epoch: {}'.format(epoch))

    torch.save(net.state_dict(), out_dir+'/swa_{}_resnet50_model.pth'.format(mode))

        
if __name__ == '__main__':
    # Hyperparameters
    
    mode = 'multimodal'
    model_name = 'FusionM4Net-FS'
    shape = (224, 224)
    batch_size = 32
    num_workers = 8
    data_mode = 'self_evaluated'
    deterministic = True
    if deterministic:
        if data_mode == 'Normal':
          random_seeds = 170
        elif data_mode == 'self_evaluated':
          random_seeds = 183
    rounds = 1
    lr = 3e-5
    epochs = 250
    swa_epoch = 50

    train_dataloader, val_dataloader = generate_dataloader(shape, batch_size, num_workers, data_mode)
    
    for i in range(rounds):
        if deterministic:
            set_seed(random_seeds + i)
      # create logger
        print(random_seeds+i)
        log, out_dir = CraateLogger(mode, model_name,i,data_mode)
        net = FusionNet(class_list).cuda()
      # create optimizer
        optimizer = optim.Adam(net.parameters(), lr=lr)
        opt = SWA(optimizer)
      # create learning schdule
        cosine_learning_schule = create_cosine_learing_schdule(epochs, lr)
        run_train(model_name,mode,i)

