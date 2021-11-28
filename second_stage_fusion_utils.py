from utils import encode_test_label,Logger,encode_meta_label
import os
import torch.nn.functional as F
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score, roc_curve,auc,average_precision_score,precision_recall_curve
import cv2
from model import FusionNet
from dependency import *
import pandas as pd
from tqdm import tqdm_notebook

def create_cosine_learing_schdule(epochs,lr):
    cosine_learning_schule = []

    for epoch in range(epochs):
        cos_inner = np.pi * (epoch % epochs)  # t - 1 is used when t has 1-based indexing.
        cos_inner /= epochs
        cos_out = np.cos(cos_inner) + 1
        final_lr = float(lr / 2 * cos_out)
        cosine_learning_schule.append(final_lr)

    return cosine_learning_schule

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
    softmax = x_exp / x_exp_row_sum
    return softmax

def find_best_threshold(net,test_index_list,df,weight_file,model_name,out_dir,mode,search_num,TTA=4,size=224,candidate_mode='CosineAnnealing'):
    os.makedirs(out_dir,exist_ok=True)
    net.set_mode('valid')

    # 7-point score
    # prob
    # 1 pigment_network
    pn_pred_list = [];
    # 2 streak
    str_pred_list = []
    # 3 pigmentation
    pig_pred_list = []
    # 4 regression structure
    rs_pred_list = []
    # 5 dots and globules
    dag_pred_list = []
    # 6 blue whitish veil l
    bwv_pred_list = []
    # 7vascular structure

    vs_pred_list = []

    # label
    # 1 pigment_network
    pn_label_list = [];
    # 2 streak
    str_label_list = []
    # 3 pigmentation
    pig_label_list = []
    # 4 regression structure
    rs_label_list = []
    # 5 dots and globules
    dag_label_list = []
    # 6 blue whitish veil l
    bwv_label_list = []
    # 7vascular structure
    vs_label_list = []

    # total_pred_list
    total_pred_list = []
    total_gt_list = []
    # total
    pred_list = [];
    gt_list = []

    acc_list = []
    corresponding_weight_list = []

    #net.load_state_dict(torch.load(weight_file))

    for index_num in tqdm(test_index_list):

        img_info = df[index_num:index_num + 1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        source_dir = '../release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir + clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

        meta_data, _,_ = encode_meta_label(img_info, index_num)
        
        if TTA == 0 :
            meta_data = torch.from_numpy(np.array([meta_data]))
        elif TTA == 3:
            meta_data = torch.from_numpy(np.array([meta_data,meta_data,meta_data]))            
        elif TTA == 4 :
            meta_data = torch.from_numpy(np.array([meta_data,meta_data,meta_data,meta_data]))
        elif TTA == 6:
            meta_data = torch.from_numpy(np.array([meta_data,meta_data,meta_data,meta_data,meta_data,meta_data]))
        
        clinic_img = cv2.resize(clinic_img, (size, size))
        clinic_img_hf = cv2.flip(clinic_img, 0)
        clinic_img_vf = cv2.flip(clinic_img, 1)
        clinic_img_vhf = cv2.flip(clinic_img, -1)
        clinic_img_90 = cv2.rotate(clinic_img,0)
        clinic_img_270 = cv2.rotate(clinic_img,2)
        


        dermoscopy_img = cv2.resize(dermoscopy_img, (size, size))
        dermoscopy_img_hf = cv2.flip(dermoscopy_img, 0)
        dermoscopy_img_vf = cv2.flip(dermoscopy_img, 1)
        dermoscopy_img_vhf = cv2.flip(dermoscopy_img, -1)
        dermoscopy_img_90 = cv2.rotate(dermoscopy_img,0)
        dermoscopy_img_270 = cv2.rotate(dermoscopy_img,2)
        
        clinic_img_total =  np.array([clinic_img])        
        dermoscopy_img_total =  np.array([dermoscopy_img])
        if TTA == 4:
            dermoscopy_img_total = np.array([dermoscopy_img,dermoscopy_img_hf,dermoscopy_img_vf,dermoscopy_img_vhf])
            clinic_img_total = np.array([clinic_img,clinic_img_hf,clinic_img_vf,clinic_img_vhf])
        elif TTA == 3:
            dermoscopy_img_total = np.array([dermoscopy_img,dermoscopy_img_hf,dermoscopy_img_vf])
            clinic_img_total = np.array([clinic_img,clinic_img_hf,clinic_img_vf])            
        elif TTA == 6:
            dermoscopy_img_total = np.array([dermoscopy_img,dermoscopy_img_hf,
                                             dermoscopy_img_vf,dermoscopy_img_vhf,dermoscopy_img_90,dermoscopy_img_270]) 
            clinic_img_total = np.array([clinic_img,clinic_img_hf,clinic_img_vf,clinic_img_vhf,clinic_img_90,clinic_img_270]) 
            
        dermoscopy_img_tensor = torch.from_numpy(np.transpose(dermoscopy_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255   
        clinic_img_tensor = torch.from_numpy(np.transpose(clinic_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255
            

        [(logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs),
         (logit_diagnosis11, logit_pn11, logit_str11, logit_pig11, logit_rs11, logit_dag11, logit_bwv11,
          logit_vs11),
         (logit_diagnosis22, logit_pn22, logit_str22, logit_pig22, logit_rs22, logit_dag22, logit_bwv22,
          logit_vs22),
         ] = net(((clinic_img_tensor).cuda(), dermoscopy_img_tensor.cuda()))

            
        pred = softmax(logit_diagnosis.detach().cpu().numpy())
        pred = np.mean(pred, 0);
        pn_pred = softmax(logit_pn.detach().cpu().numpy())
        pn_pred = np.mean(pn_pred, 0);
        str_pred = softmax(logit_str.detach().cpu().numpy());
        str_pred = np.mean(str_pred, 0);
        pig_pred = softmax(logit_pig.detach().cpu().numpy());
        pig_pred = np.mean(pig_pred, 0);
        rs_pred = softmax(logit_rs.detach().cpu().numpy());
        rs_pred = np.mean(rs_pred, 0);
        dag_pred = softmax(logit_dag.detach().cpu().numpy());
        dag_pred = np.mean(dag_pred, 0);
        bwv_pred = softmax(logit_bwv.detach().cpu().numpy());
        bwv_pred = np.mean(bwv_pred, 0);
        vs_pred = softmax(logit_vs.detach().cpu().numpy());
        vs_pred = np.mean(vs_pred, 0);

        pred11 = softmax(logit_diagnosis11.detach().cpu().numpy())
        pred11 = np.mean(pred11, 0);
        pn_pred11 = softmax(logit_pn11.detach().cpu().numpy())
        pn_pred11 = np.mean(pn_pred11, 0);
        str_pred11 = softmax(logit_str11.detach().cpu().numpy());
        str_pred11 = np.mean(str_pred11, 0);
        pig_pred11 = softmax(logit_pig11.detach().cpu().numpy());
        pig_pred11 = np.mean(pig_pred11, 0);
        rs_pred11 = softmax(logit_rs11.detach().cpu().numpy());
        rs_pred11 = np.mean(rs_pred11, 0);
        dag_pred11 = softmax(logit_dag11.detach().cpu().numpy());
        dag_pred11 = np.mean(dag_pred11, 0);
        bwv_pred11 = softmax(logit_bwv11.detach().cpu().numpy());
        bwv_pred11 = np.mean(bwv_pred11, 0);
        vs_pred11 = softmax(logit_vs11.detach().cpu().numpy());
        vs_pred11 = np.mean(vs_pred11, 0);

        pred22 = softmax(logit_diagnosis22.detach().cpu().numpy())
        pred22 = np.mean(pred22, 0);
        pn_pred22 = softmax(logit_pn22.detach().cpu().numpy())
        pn_pred22 = np.mean(pn_pred22, 0);
        str_pred22 = softmax(logit_str22.detach().cpu().numpy());
        str_pred22 = np.mean(str_pred22, 0);
        pig_pred22 = softmax(logit_pig22.detach().cpu().numpy());
        pig_pred22 = np.mean(pig_pred22, 0);
        rs_pred22 = softmax(logit_rs22.detach().cpu().numpy());
        rs_pred22 = np.mean(rs_pred22, 0);
        dag_pred22 = softmax(logit_dag22.detach().cpu().numpy());
        dag_pred22 = np.mean(dag_pred22, 0);
        bwv_pred22 = softmax(logit_bwv22.detach().cpu().numpy());
        bwv_pred22 = np.mean(bwv_pred22, 0);
        vs_pred22 = softmax(logit_vs22.detach().cpu().numpy());
        vs_pred22 = np.mean(vs_pred22, 0);


        total_pred_list.append([(logit_diagnosis.detach().cpu().numpy(), logit_pn.detach().cpu().numpy(), logit_str.detach().cpu().numpy(),
                                 logit_pig.detach().cpu().numpy(), logit_rs.detach().cpu().numpy(), logit_dag.detach().cpu().numpy(),
                                 logit_bwv.detach().cpu().numpy(), logit_vs.detach().cpu().numpy()),
         (logit_diagnosis11.detach().cpu().numpy(), logit_pn11.detach().cpu().numpy(), logit_str11.detach().cpu().numpy(),
          logit_pig11.detach().cpu().numpy(), logit_rs11.detach().cpu().numpy(), logit_dag11.detach().cpu().numpy(),
          logit_bwv11.detach().cpu().numpy(),logit_vs11.detach().cpu().numpy()),
         (logit_diagnosis22.detach().cpu().numpy(), logit_pn22.detach().cpu().numpy(), logit_str22.detach().cpu().numpy(),
          logit_pig22.detach().cpu().numpy(), logit_rs22.detach().cpu().numpy(), logit_dag22.detach().cpu().numpy(),
          logit_bwv22.detach().cpu().numpy(),   logit_vs22.detach().cpu().numpy())
         ])


        [diagnosis_label, pigment_network_label, streaks_label, pigmentation_label, regression_structures_label,
         dots_and_globules_label, blue_whitish_veil_label, vascular_structures_label], _ = encode_test_label(img_info, index_num)

        # diagnositic_label
        gt_list.append(diagnosis_label)
        # pn_label
        pn_label_list.append(pigment_network_label)
        # str_label
        str_label_list.append(streaks_label)
        # pig_label
        pig_label_list.append(pigmentation_label)
        # rs_label
        rs_label_list.append(regression_structures_label)
        # dag_label
        dag_label_list.append(dots_and_globules_label)
        # bwv_label
        bwv_label_list.append(blue_whitish_veil_label)
        # vs_label
        vs_label_list.append(vascular_structures_label)

        gt = np.array(gt_list)
        pn_gt = np.array(pn_label_list)
        str_gt = np.array(str_label_list)
        pig_gt = np.array(pig_label_list)
        rs_gt = np.array(rs_label_list)
        dag_gt = np.array(dag_label_list)
        bwv_gt = np.array(bwv_label_list)
        vs_gt = np.array(vs_label_list)

    logit_diagnosis,logit_diagnosis11,logit_diagnosis22 = [],[],[]
    logit_pn,logit_pn11,logit_pn22 = [],[],[]
    logit_str,logit_str11,logit_str22= [],[],[]
    logit_pig,logit_pig11,logit_pig22 = [],[],[]
    logit_rs,logit_rs11,logit_rs22 = [],[],[]
    logit_dag,logit_dag11,logit_dag22 = [],[],[]
    logit_bwv,logit_bwv11,logit_bwv22 = [],[],[]
    logit_vs,logit_vs11,logit_vs22 = [],[],[]

    for single_pred in total_pred_list:

        logit_diagnosis.append(single_pred[0][0])
        logit_pn.append(single_pred[0][1])
        logit_str.append(single_pred[0][2])
        logit_pig.append(single_pred[0][3])
        logit_rs.append(single_pred[0][4])
        logit_dag.append(single_pred[0][5])
        logit_bwv.append(single_pred[0][6])
        logit_vs.append(single_pred[0][7])

        logit_diagnosis11.append(single_pred[1][0])
        logit_pn11.append(single_pred[1][1])
        logit_str11.append(single_pred[1][2])
        logit_pig11.append(single_pred[1][3])
        logit_rs11.append(single_pred[1][4])
        logit_dag11.append(single_pred[1][5])
        logit_bwv11.append(single_pred[1][6])
        logit_vs11.append(single_pred[1][7])

        logit_diagnosis22.append(single_pred[2][0])
        logit_pn22.append(single_pred[2][1])
        logit_str22.append(single_pred[2][2])
        logit_pig22.append(single_pred[2][3])
        logit_rs22.append(single_pred[2][4])
        logit_dag22.append(single_pred[2][5])
        logit_bwv22.append(single_pred[2][6])
        logit_vs22.append(single_pred[2][7])

    logit_diagnosis = np.array(logit_diagnosis)
    logit_pn = np.array(logit_pn)
    logit_str= np.array(logit_str)
    logit_pig= np.array(logit_pig)
    logit_rs = np.array(logit_rs)
    logit_dag= np.array(logit_dag)
    logit_bwv= np.array(logit_bwv)
    logit_vs= np.array(logit_vs)

    logit_diagnosis11 = np.array(logit_diagnosis11)
    logit_pn11 = np.array(logit_pn11)
    logit_str11= np.array(logit_str11)
    logit_pig11= np.array(logit_pig11)
    logit_rs11 = np.array(logit_rs11)
    logit_dag11= np.array(logit_dag11)
    logit_bwv11= np.array(logit_bwv11)
    logit_vs11= np.array(logit_vs11)

    logit_diagnosis22 = np.array(logit_diagnosis22)
    logit_pn22 = np.array(logit_pn22)
    logit_str22= np.array(logit_str22)
    logit_pig22= np.array(logit_pig22)
    logit_rs22 = np.array(logit_rs22)
    logit_dag22= np.array(logit_dag22)
    logit_bwv22= np.array(logit_bwv22)
    logit_vs22= np.array(logit_vs22)

    best_acc = 0
    best_weight = 0
    acc_list = []
    save_weight_list = []
    if candidate_mode == 'CosineAnnealing':
      weight_list  = create_cosine_learing_schdule(search_num,1)
    elif candidate_mode == 'Linear':
      weight_list  = np.linspace(0.05,1,num=search_num)
    print(weight_list)

    for weight in tqdm(weight_list):
        for weight_1 in weight_list:
           for weight_2 in weight_list:

                    weight_total = weight + weight_1 + weight_2
                    if weight_total > 1:  # Note that these two command lines are updated after the paper publish since we just got the idea recenly to skip lots of repeted weighted array.
                                          # so we didn't write it in the pseudo code of Algorithm. 1 of our paper. 
                        continue          # There is no big difference on performance between the model using or not using these two command lines.
                                          # But it significantly speed up the searching process. So, we suggest you use it and when you use and set the search_num as 100, it just take within 10 second.
                    weight = weight / weight_total
                    weight_1 = weight_1 / weight_total
                    weight_2 = weight_2 / weight_total
                    

                    logit_diagnosis_total = ((weight_1)*logit_diagnosis11 + weight*logit_diagnosis + weight_2*logit_diagnosis22)
                    logit_pn_total = ((weight_1)*logit_pn11 + weight*logit_pn + weight_2*logit_pn22 )
                    logit_str_total = ((weight_1)*logit_str11 + weight*logit_str + weight_2*logit_str22)
                    logit_pig_total = ((weight_1)*logit_pig11 + weight*logit_pig + weight_2*logit_pig22)
                    logit_rs_total = ((weight_1)*logit_rs11 + weight*logit_rs + weight_2*logit_rs22)
                    logit_dag_total = ((weight_1)*logit_dag11 + weight*logit_dag + weight_2*logit_dag22)
                    logit_bwv_total = ((weight_1)*logit_bwv11 + weight*logit_bwv + weight_2*logit_bwv22)
                    logit_vs_total = ((weight_1)*logit_vs11 + weight*logit_vs + weight_2*logit_vs22)


                    pred_list = []
                    pn_pred_list = []
                    str_pred_list = []
                    pig_pred_list = []
                    rs_pred_list = []
                    dag_pred_list = []
                    bwv_pred_list = []
                    vs_pred_list = []

                    for num in range(logit_diagnosis_total.shape[0]):
                        logit_diagnosis_sub = logit_diagnosis_total[num]
                        logit_pn_sub = logit_pn_total[num]
                        logit_str_sub = logit_str_total[num]
                        logit_pig_sub = logit_pig_total[num]
                        logit_rs_sub = logit_rs_total[num]
                        logit_dag_sub = logit_dag_total[num]
                        logit_bwv_sub = logit_bwv_total[num]
                        logit_vs_sub = logit_vs_total[num]

                        pred = softmax(logit_diagnosis_sub)
                        pred = np.mean(pred, 0);
                        pn_pred = softmax(logit_pn_sub)
                        pn_pred = np.mean(pn_pred, 0);
                        str_pred = softmax(logit_str_sub);
                        str_pred = np.mean(str_pred, 0);
                        pig_pred = softmax(logit_pig_sub);
                        pig_pred = np.mean(pig_pred, 0);
                        rs_pred = softmax(logit_rs_sub);
                        rs_pred = np.mean(rs_pred, 0);
                        dag_pred = softmax(logit_dag_sub);
                        dag_pred = np.mean(dag_pred, 0);
                        bwv_pred = softmax(logit_bwv_sub);
                        bwv_pred = np.mean(bwv_pred, 0);
                        vs_pred = softmax(logit_vs_sub);
                        vs_pred = np.mean(vs_pred, 0);

                        pred_ = np.argmax(pred)
                        pn_pred_ = np.argmax(pn_pred)
                        str_pred_ = np.argmax(str_pred)
                        pig_pred_ = np.argmax(pig_pred)
                        rs_pred_ = np.argmax(rs_pred)
                        dag_pred_ = np.argmax(dag_pred)
                        bwv_pred_ = np.argmax(bwv_pred)
                        vs_pred_ = np.argmax(vs_pred)

                        pred_list.append(pred_)
                        pn_pred_list.append(pn_pred_);
                        str_pred_list.append(str_pred_);
                        pig_pred_list.append(pig_pred_);
                        rs_pred_list.append(rs_pred_);
                        dag_pred_list.append(dag_pred_);
                        bwv_pred_list.append(bwv_pred_);
                        vs_pred_list.append(vs_pred_);
                    #print(vs_pred)
                    pred = np.array(pred_list).squeeze();
                    pn_pred = np.array(pn_pred_list).squeeze();
                    str_pred = np.array(str_pred_list).squeeze();
                    pig_pred = np.array(pig_pred_list).squeeze();
                    rs_pred = np.array(rs_pred_list).squeeze();
                    dag_pred = np.array(dag_pred_list).squeeze();
                    bwv_pred = np.array(bwv_pred_list).squeeze();
                    vs_pred = np.array(vs_pred_list).squeeze();

                    #print(vs_pred.shape,vs_gt.shape)
                    vs_acc = np.mean(vs_pred == vs_gt)
                    bwv_acc = np.mean(bwv_pred == bwv_gt)
                    dag_acc = np.mean(dag_pred == dag_gt)
                    rs_acc = np.mean(rs_pred == rs_gt)
                    pig_acc = np.mean(pig_pred == pig_gt)
                    str_acc = np.mean(str_pred == str_gt)
                    pn_acc = np.mean(pn_pred == pn_gt)
                    diag_acc = np.mean(pred == gt)

                    avg_acc = (vs_acc + bwv_acc + dag_acc + rs_acc + pig_acc + str_acc + pn_acc + diag_acc) / 8
                   # print(avg_acc)
                   # print(weight_2,weight,weight_1)
                    #acc_list.append(avg_acc)
                    #save_weight_list.append([weight_2,weight,weight_1])
                    if avg_acc > best_acc :
                        best_acc =avg_acc
                        best_weight = (weight_2,weight,weight_1)
                        print(best_acc)
                        print(best_weight)
                      #  acc_list.append(avg_acc)
                      #  corresponding_weight_list.append((weight,weight_1))
                        #print(max(acc_list))
    return acc_list,save_weight_list, best_weight

def predict(net,test_index_list,df,weight_file,model_name,out_dir,mode,weight_list,TTA=4,size=224):

    os.makedirs(out_dir,exist_ok=True)
    log = Logger()
    log.open(out_dir + 'log.single_modality_{}_{}_skinlesion.txt'.format(mode,model_name), mode='w')
    log.write('\n--- [START %s] %s\n\n' % ('IDENTIFIER', '-' * 64))
    net.set_mode('valid')

    # 7-point score
    # prob
    # 1 pigment_network
    pn_prob_typ_list = [];
    pn_prob_asp_list = [];
    pn_pred_list = [];
    pn_prob_asb_list = [];
    pn_prob_list = []
    # 2 streak
    str_prob_asb_list = [];
    str_prob_reg_list = [];
    str_prob_irg_list = [];
    str_pred_list = [];
    str_prob_list = []
    
    # 3 pigmentation
    pig_prob_asb_list = [];
    pig_prob_reg_list = [];
    pig_prob_irg_list = [];
    pig_pred_list = []
    pig_prob_list = []
    
    # 4 regression structure
    rs_prob_asb_list = [];
    rs_prob_prs_list = [];
    rs_pred_list = []
    rs_prob_list = []
    
    # 5 dots and globules
    dag_prob_asb_list = [];
    dag_prob_reg_list = [];
    dag_prob_irg_list = [];
    dag_pred_list = []
    dag_prob_list = []
    
    # 6 blue whitish veil l
    bwv_prob_asb_list = [];
    bwv_prob_prs_list = [];
    bwv_pred_list = []
    bwv_prob_list = []
    
    # 7vascular structure
    vs_prob_asb_list = [];
    vs_prob_reg_list = [];
    vs_prob_irg_list = [];
    vs_pred_list = []
    vs_prob_list = []

    # label
    # 1 pigment_network
    pn_label_typ_list = [];
    pn_label_asp_list = [];
    pn_label_list = [];
    pn_label_asb_list = []
    # 2 streak
    str_label_asb_list = [];
    str_label_reg_list = [];
    str_label_irg_list = [];
    str_label_list = []
    # 3 pigmentation
    pig_label_asb_list = [];
    pig_label_reg_list = [];
    pig_label_irg_list = [];
    pig_label_list = []
    # 4 regression structure
    rs_label_asb_list = [];
    rs_label_prs_list = [];
    rs_label_list = []
    # 5 dots and globules
    dag_label_asb_list = [];
    dag_label_reg_list = [];
    dag_label_irg_list = [];
    dag_label_list = []
    # 6 blue whitish veil l
    bwv_label_asb_list = [];
    bwv_label_prs_list = [];
    bwv_label_list = []
    # 7vascular structure
    vs_label_asb_list = [];
    vs_label_reg_list = [];
    vs_label_irg_list = [];
    vs_label_list = []

    # total
    pred_list = [];
    prob_list = [];
    gt_list = []

    # diagnositi_prob and diagnositic_label
    nevu_prob_list = [];
    bcc_prob_list = [];
    mel_prob_list = [];
    misc_prob_list = [];
    sk_prob_list = []
    nevu_label_list = [];
    bcc_label_list = [];
    mel_label_list = [];
    misc_label_list = [];
    sk_label_list = []
    seven_point_feature_list = []


    for index_num in tqdm(test_index_list):
        img_info = df[index_num:index_num + 1]
        clinic_path = img_info['clinic']
        dermoscopy_path = img_info['derm']
        source_dir = '../release_v0/release_v0/images/'
        clinic_img = cv2.imread(source_dir + clinic_path[index_num])
        dermoscopy_img = cv2.imread(source_dir + dermoscopy_path[index_num])

        meta_data, _,_ = encode_meta_label(img_info, index_num)
        
        if TTA == 0 :
            meta_data = torch.from_numpy(np.array([meta_data]))
        elif TTA ==4 :
            meta_data = torch.from_numpy(np.array([meta_data,meta_data,meta_data,meta_data]))
        elif TTA == 6:
            meta_data = torch.from_numpy(np.array([meta_data,meta_data,meta_data,meta_data,meta_data,meta_data]))
        
        clinic_img = cv2.resize(clinic_img, (size, size))
        clinic_img_hf = cv2.flip(clinic_img, 0)
        clinic_img_vf = cv2.flip(clinic_img, 1)
        clinic_img_vhf = cv2.flip(clinic_img, -1)
        clinic_img_90 = cv2.rotate(clinic_img,0)
        clinic_img_270 = cv2.rotate(clinic_img,2)
        
        dermoscopy_img = cv2.resize(dermoscopy_img, (size, size))
        dermoscopy_img_hf = cv2.flip(dermoscopy_img, 0)
        dermoscopy_img_vf = cv2.flip(dermoscopy_img, 1)
        dermoscopy_img_vhf = cv2.flip(dermoscopy_img, -1)
        dermoscopy_img_90 = cv2.rotate(dermoscopy_img,0)
        dermoscopy_img_270 = cv2.rotate(dermoscopy_img,2)
        
        clinic_img_total =  np.array([clinic_img])        
        dermoscopy_img_total =  np.array([dermoscopy_img])
        if TTA == 4:
            dermoscopy_img_total = np.array([dermoscopy_img,dermoscopy_img_hf,dermoscopy_img_vf,dermoscopy_img_vhf])
            clinic_img_total = np.array([clinic_img,clinic_img_hf,clinic_img_vf,clinic_img_vhf])
        elif TTA == 6:
            dermoscopy_img_total = np.array([dermoscopy_img,dermoscopy_img_hf,
                                             dermoscopy_img_vf,dermoscopy_img_vhf,dermoscopy_img_90,dermoscopy_img_270]) 
            clinic_img_total = np.array([clinic_img,clinic_img_hf,clinic_img_vf,clinic_img_vhf,clinic_img_90,clinic_img_270]) 
            
        dermoscopy_img_tensor = torch.from_numpy(np.transpose(dermoscopy_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255   
        clinic_img_tensor = torch.from_numpy(np.transpose(clinic_img_total, [0, 3, 1, 2]).astype(np.float32)) / 255


        

        [(logit_diagnosis, logit_pn, logit_str, logit_pig, logit_rs, logit_dag, logit_bwv, logit_vs),
         (logit_diagnosis11, logit_pn11, logit_str11, logit_pig11, logit_rs11, logit_dag11, logit_bwv11,
          logit_vs11),
         (logit_diagnosis22, logit_pn22, logit_str22, logit_pig22, logit_rs22, logit_dag22, logit_bwv22,
          logit_vs22),
         ] = net(
            ((clinic_img_tensor).cuda(), dermoscopy_img_tensor.cuda()))

        
        weight = weight_list[0]
        weight_1 = weight_list[1]
        weight_2 = weight_list[2]

        logit_diagnosis = (weight_1*logit_diagnosis11 + weight*logit_diagnosis + weight_2*logit_diagnosis22)
        logit_pn = (weight_1*logit_pn11 + weight*logit_pn + weight_2*logit_pn22 )
        logit_str = (weight_1*logit_str11 + weight*logit_str + weight_2*logit_str22)
        logit_pig = (weight_1*logit_pig11 + weight*logit_pig + weight_2*logit_pig22)
        logit_rs = (weight_1*logit_rs11 + weight*logit_rs + weight_2*logit_rs22)
        logit_dag = (weight_1*logit_dag11 + weight*logit_dag + weight_2*logit_dag22)
        logit_bwv = (weight_1*logit_bwv11 + weight*logit_bwv + weight_2*logit_bwv22)
        logit_vs = (weight_1*logit_vs11 + weight*logit_vs + weight_2*logit_vs22)

      
        # diagnositic_pred
        pred = softmax(logit_diagnosis.detach().cpu().numpy());
        pred = np.mean(pred, 0);
        pred_ = np.argmax(pred)
        nevu_prob = pred[0];
        bcc_prob = pred[1];
        mel_prob = pred[2];
        misc_prob = pred[3];
        sk_prob = pred[4]
        # pn_prob
        pn_pred = softmax(logit_pn.detach().cpu().numpy());
        pn_pred = np.mean(pn_pred, 0);
        pn_pred_ = np.argmax(pn_pred)
        pn_prob_asb = pn_pred[0];
        pn_prob_typ = pn_pred[1];
        pn_prob_asp = pn_pred[2];

        # str_prob
        str_pred = softmax(logit_str.detach().cpu().numpy());
        str_pred = np.mean(str_pred, 0);
        str_pred_ = np.argmax(str_pred)
        str_prob_asb = str_pred[0];
        str_prob_reg = str_pred[1];
        str_prob_irg = str_pred[2]
        # pig_prob
        pig_pred = softmax(logit_pig.detach().cpu().numpy());
        pig_pred = np.mean(pig_pred, 0);
        pig_pred_ = np.argmax(pig_pred)
        pig_prob_asb = pig_pred[0];
        pig_prob_reg = pig_pred[1];
        pig_prob_irg = pig_pred[2]
        # rs_prob
        rs_pred = softmax(logit_rs.detach().cpu().numpy());
        rs_pred = np.mean(rs_pred, 0);
        rs_pred_ = np.argmax(rs_pred)
        
        rs_prob_asb = rs_pred[0];
        rs_prob_prs = rs_pred[1]
        # dag_prob
        dag_pred = softmax(logit_dag.detach().cpu().numpy());
        dag_pred = np.mean(dag_pred, 0);
        dag_pred_ = np.argmax(dag_pred)
        dag_prob_asb = dag_pred[0];
        dag_prob_reg = dag_pred[1];
        dag_prob_irg = dag_pred[2]
        # bwv_prob
        bwv_pred = softmax(logit_bwv.detach().cpu().numpy());
        bwv_pred = np.mean(bwv_pred, 0);
        bwv_pred_ = np.argmax(bwv_pred)
        bwv_prob_asb = bwv_pred[0];
        bwv_prob_prs = bwv_pred[1]
        # vs_prob
        vs_pred = softmax(logit_vs.detach().cpu().numpy());
        vs_pred = np.mean(vs_pred, 0);
        vs_pred_ = np.argmax(vs_pred)
        vs_prob_asb = vs_pred[0];
        vs_prob_reg = vs_pred[1];
        vs_prob_irg = vs_pred[2]

        seven_point_feature_list.append(np.concatenate([pred, pn_pred,str_pred,pig_pred
                                                        ,rs_pred,dag_pred,bwv_pred,vs_pred],0))
        # encode label
        [diagnosis_label, pigment_network_label, streaks_label, pigmentation_label, regression_structures_label,
         dots_and_globules_label, blue_whitish_veil_label, vascular_structures_label], [diagnosis_label_one_hot,
                                                                                        pigment_network_label_one_hot,
                                                                                        streaks_label_one_hot,
                                                                                        pigmentation_label_one_hot,
                                                                                        regression_structures_label_one_hot,
                                                                                        dots_and_globules_label_one_hot,
                                                                                        blue_whitish_veil_label_one_hot,
                                                                                        vascular_structures_label_one_hot] = encode_test_label(img_info, index_num)

        # diagnositic_label
        pred_list.append(pred_);
        prob_list.append(pred);
        
        gt_list.append(diagnosis_label)
        nevu_prob_list.append(nevu_prob);
        bcc_prob_list.append(bcc_prob);
        mel_prob_list.append(mel_prob);
        misc_prob_list.append(misc_prob);
        sk_prob_list.append(sk_prob)
        nevu_label_list.append(diagnosis_label_one_hot[0]);
        bcc_label_list.append(diagnosis_label_one_hot[1]);
        mel_label_list.append(diagnosis_label_one_hot[2]);
        misc_label_list.append(diagnosis_label_one_hot[3]);
        sk_label_list.append(diagnosis_label_one_hot[4])

        # pn_label
        pn_pred_list.append(pn_pred_);
        pn_prob_list.append(pn_pred);
        
        pn_label_list.append(pigment_network_label)
        pn_prob_typ_list.append(pn_prob_typ);
        pn_prob_asp_list.append(pn_prob_asp);
        pn_prob_asb_list.append(pn_prob_asb);

        pn_label_asb_list.append(pigment_network_label_one_hot[0]);
        pn_label_typ_list.append(pigment_network_label_one_hot[1]);
        pn_label_asp_list.append(pigment_network_label_one_hot[2]);


        # str_label
        str_pred_list.append(str_pred_);
        str_prob_list.append(str_pred);
        
        str_label_list.append(streaks_label)
        str_prob_reg_list.append(str_prob_reg);
        str_prob_irg_list.append(str_prob_irg);
        str_prob_asb_list.append(str_prob_asb)
        str_label_asb_list.append(streaks_label_one_hot[0]);
        str_label_reg_list.append(streaks_label_one_hot[1]);
        str_label_irg_list.append(streaks_label_one_hot[2])

        # pig_label
        pig_pred_list.append(pig_pred_);
        pig_prob_list.append(pig_pred);
        
        
        pig_label_list.append(pigmentation_label)
        pig_prob_reg_list.append(pig_prob_reg);
        pig_prob_irg_list.append(pig_prob_irg);
        pig_prob_asb_list.append(pig_prob_asb)
        pig_label_asb_list.append(pigmentation_label_one_hot[0]);
        pig_label_reg_list.append(pigmentation_label_one_hot[1]);
        pig_label_irg_list.append(pigmentation_label_one_hot[2])

        # rs_label
        rs_pred_list.append(rs_pred_);
        rs_prob_list.append(rs_pred);
        
        rs_label_list.append(regression_structures_label)
        rs_prob_asb_list.append(rs_prob_asb);
        rs_prob_prs_list.append(rs_prob_prs)
        rs_label_asb_list.append(regression_structures_label_one_hot[0]);
        rs_label_prs_list.append(regression_structures_label_one_hot[1])

        # dag_label
        dag_pred_list.append(dag_pred_);
        dag_prob_list.append(dag_pred);
        
        dag_label_list.append(dots_and_globules_label)
        dag_prob_reg_list.append(dag_prob_reg);
        dag_prob_irg_list.append(dag_prob_irg);
        dag_prob_asb_list.append(dag_prob_asb)
        dag_label_asb_list.append(dots_and_globules_label_one_hot[0]);
        dag_label_reg_list.append(dots_and_globules_label_one_hot[1]);
        dag_label_irg_list.append(dots_and_globules_label_one_hot[2])

        # bwv_label
        bwv_pred_list.append(bwv_pred_);
        bwv_prob_list.append(bwv_pred);
        
        bwv_label_list.append(blue_whitish_veil_label)
        bwv_prob_asb_list.append(bwv_prob_asb);
        bwv_prob_prs_list.append((bwv_prob_prs))
        bwv_label_asb_list.append(blue_whitish_veil_label_one_hot[0]);
        bwv_label_prs_list.append(blue_whitish_veil_label_one_hot[1])

        # vs_label
        vs_pred_list.append(vs_pred_);
        vs_prob_list.append(vs_pred);
        
        vs_label_list.append(vascular_structures_label)
        vs_prob_reg_list.append(vs_prob_reg);
        vs_prob_irg_list.append(vs_prob_irg);
        vs_prob_asb_list.append(vs_prob_asb)
        vs_label_asb_list.append(vascular_structures_label_one_hot[0]);
        vs_label_reg_list.append(vascular_structures_label_one_hot[1]);
        vs_label_irg_list.append(vascular_structures_label_one_hot[2])

    pred = np.array(pred_list).squeeze();
    prob = np.array(prob_list).squeeze();
    
    gt = np.array(gt_list)
    nevu_prob = np.array(nevu_prob_list);
    bcc_prob = np.array(bcc_prob_list);
    mel_prob = np.array(mel_prob_list);
    misc_prob = np.array(misc_prob_list);
    sk_prob = np.array(sk_prob_list)
    nevu_label = np.array(nevu_label_list);
    bcc_label = np.array(bcc_label_list);
    mel_label = np.array(mel_label_list);
    misc_label = np.array(misc_label_list);
    sk_label = np.array(sk_label_list)

    pn_pred = np.array(pn_pred_list).squeeze();
    pn_prob = np.array(pn_prob_list).squeeze();
    
    pn_gt = np.array(pn_label_list)
    pn_prob_typ = np.array(pn_prob_typ_list);
    pn_prob_asp = np.array(pn_prob_asp_list);
    pn_prob_asb = np.array(pn_prob_asb_list)

    pn_label_typ = np.array(pn_label_typ_list);
    pn_label_asp = np.array(pn_label_asp_list);
    pn_label_asb = np.array(pn_label_asb_list)

    str_pred = np.array(str_pred_list).squeeze();
    str_prob = np.array(str_prob_list).squeeze();
    
    str_gt = np.array(str_label_list)
    str_prob_asb = np.array(str_prob_asb_list);
    str_prob_reg = np.array(str_prob_reg_list);
    str_prob_irg = np.array(str_prob_irg_list)
    str_label_asb = np.array(str_label_asb_list);
    str_label_reg = np.array(str_label_reg_list);
    str_label_irg = np.array(str_label_irg_list)

    pig_pred = np.array(pig_pred_list).squeeze();
    pig_prob = np.array(pig_prob_list).squeeze();

    pig_gt = np.array(pig_label_list)
    pig_prob_asb = np.array(pig_prob_asb_list);
    pig_prob_reg = np.array(pig_prob_reg_list);
    pig_prob_irg = np.array(pig_prob_irg_list)
    pig_label_asb = np.array(pig_label_asb_list);
    pig_label_reg = np.array(pig_label_reg_list);
    pig_label_irg = np.array(pig_label_irg_list)

    rs_pred = np.array(rs_pred_list).squeeze();
    rs_prob = np.array(rs_prob_list).squeeze();
    
    rs_gt = np.array(rs_label_list)
    rs_prob_asb = np.array(rs_prob_asb_list);
    rs_prob_prs = np.array(rs_prob_prs_list)
    rs_label_asb = np.array(rs_label_asb_list);
    rs_label_prs = np.array(rs_label_prs_list)

    dag_pred = np.array(dag_pred_list).squeeze();
    dag_prob = np.array(dag_prob_list).squeeze();
    
    dag_gt = np.array(dag_label_list)
    dag_prob_asb = np.array(dag_prob_asb_list);
    dag_prob_reg = np.array(dag_prob_reg_list);
    dag_prob_irg = np.array(dag_prob_irg_list)
    dag_label_asb = np.array(dag_label_asb_list);
    dag_label_reg = np.array(dag_label_reg_list);
    dag_label_irg = np.array(dag_label_irg_list)

    bwv_pred = np.array(bwv_pred_list).squeeze();
    bwv_prob = np.array(bwv_prob_list).squeeze();
    
    bwv_gt = np.array(bwv_label_list)
    bwv_prob_asb = np.array(bwv_prob_asb_list);
    bwv_prob_prs = np.array(bwv_prob_prs_list)
    bwv_label_asb = np.array(bwv_label_asb_list);
    bwv_label_prs = np.array(bwv_label_prs_list)

    vs_pred = np.array(vs_pred_list).squeeze();
    vs_prob = np.array(vs_prob_list).squeeze();
        
    vs_gt = np.array(vs_label_list)
    vs_prob_asb = np.array(vs_prob_asb_list);
    vs_prob_reg = np.array(vs_prob_reg_list);
    vs_prob_irg = np.array(vs_prob_irg_list)
    vs_label_asb = np.array(vs_label_asb_list);
    vs_label_reg = np.array(vs_label_reg_list);
    vs_label_irg = np.array(vs_label_irg_list)

    vs_acc = np.mean(vs_pred == vs_gt)
    bwv_acc = np.mean(bwv_pred == bwv_gt)
    dag_acc = np.mean(dag_pred == dag_gt)
    rs_acc = np.mean(rs_pred == rs_gt)
    pig_acc = np.mean(pig_pred == pig_gt)
    str_acc = np.mean(str_pred == str_gt)
    pn_acc = np.mean(pn_pred == pn_gt)
    diag_acc = np.mean(pred == gt)


    avg_acc = (vs_acc + bwv_acc + dag_acc + rs_acc + pig_acc + str_acc + pn_acc + diag_acc) / 8
    log.write('-'*15 + '\n')
    log.write('avg_acc : {}\n'.format(avg_acc))
    log.write('vs_acc : {}\n'.format(np.mean(vs_pred == vs_gt)))
    log.write('bwv_acc : {}\n'.format(np.mean(bwv_pred == bwv_gt)))
    log.write('dag_acc : {}\n'.format(np.mean(dag_pred == dag_gt)))
    log.write('rs_acc : {}\n'.format(np.mean(rs_pred == rs_gt)))
    log.write('pig_acc : {}\n'.format(np.mean(pig_pred == pig_gt)))
    log.write('str_acc : {}\n'.format(np.mean(str_pred == str_gt)))
    log.write('pn_acc : {}\n'.format(np.mean(pn_pred == pn_gt)))
    log.write('diag_acc : {}\n'.format(np.mean(pred == gt)))


    nevu_auc = roc_auc_score((np.array(nevu_label)*1).flatten(),nevu_prob.flatten())
    bcc_auc = roc_auc_score((np.array(bcc_label)*1).flatten(),bcc_prob.flatten())
    mel_auc = roc_auc_score((np.array(mel_label)*1).flatten(),mel_prob.flatten())
    misc_auc = roc_auc_score((np.array(misc_label)*1).flatten(),misc_prob.flatten())
    sk_auc = roc_auc_score((np.array(sk_label)*1).flatten(),sk_prob.flatten())
    log.write('-' * 15 + "\n")
    log.write('nevu_auc: {}\n'.format(nevu_auc))
    log.write('bcc_auc: {}\n'.format(bcc_auc))
    log.write('mel_auc: {}\n'.format(mel_auc))
    log.write('misc_auc: {}\n'.format(misc_auc))
    log.write('sk_auc: {}\n'.format(sk_auc))

    vs_asb_auc = roc_auc_score((np.array(vs_label_asb) * 1).flatten(), vs_prob_asb.flatten())
    vs_reg_auc = roc_auc_score((np.array(vs_label_reg) * 1).flatten(), vs_prob_reg.flatten())
    vs_irg_auc = roc_auc_score((np.array(vs_label_irg) * 1).flatten(), vs_prob_irg.flatten())
    log.write('-' * 15 + "\n")
    log.write('vs_asb_auc: {}\n'.format(vs_asb_auc))
    log.write('vs_reg_auc: {}\n'.format(vs_reg_auc))
    log.write('vs_irg_auc: {}\n'.format(vs_irg_auc))

    bwv_asb_auc = roc_auc_score((np.array(bwv_label_asb) * 1).flatten(), bwv_prob_asb.flatten())
    bwv_prs_auc = roc_auc_score((np.array(bwv_label_prs) * 1).flatten(), bwv_prob_prs.flatten())
    log.write('-' * 15 + '\n')
    log.write('bwv_asb_auc: {}\n'.format(bwv_asb_auc))
    log.write('bwv_prs_auc: {}\n'.format(bwv_prs_auc))

    dag_asb_auc = roc_auc_score((np.array(dag_label_asb) * 1).flatten(), dag_prob_asb.flatten())
    dag_reg_auc = roc_auc_score((np.array(dag_label_reg) * 1).flatten(), dag_prob_reg.flatten())
    dag_irg_auc = roc_auc_score((np.array(dag_label_irg) * 1).flatten(), dag_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('dag_asb_auc: {}\n'.format(dag_asb_auc))
    log.write('dag_reg_auc: {}\n'.format(dag_reg_auc))
    log.write('dag_irg_auc: {}\n'.format(dag_irg_auc))

    rs_asb_auc = roc_auc_score((np.array(rs_label_asb) * 1).flatten(), rs_prob_asb.flatten())
    rs_prs_auc = roc_auc_score((np.array(rs_label_prs) * 1).flatten(), rs_prob_prs.flatten())
    log.write('-' * 15 + '\n')
    log.write('rs_asb_auc: {}\n'.format(rs_asb_auc))
    log.write('rs_prs_auc: {}\n'.format(rs_prs_auc))

    pig_asb_auc = roc_auc_score((np.array(pig_label_asb) * 1).flatten(), pig_prob_asb.flatten())
    pig_reg_auc = roc_auc_score((np.array(pig_label_reg) * 1).flatten(), pig_prob_reg.flatten())
    pig_irg_auc = roc_auc_score((np.array(pig_label_irg) * 1).flatten(), pig_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('pig_asb_auc: {}\n'.format(pig_asb_auc))
    log.write('pig_reg_auc: {}\n'.format(pig_reg_auc))
    log.write('pig_irg_auc: {}\n'.format(pig_irg_auc))

    str_asb_auc = roc_auc_score((np.array(str_label_asb) * 1).flatten(), str_prob_asb.flatten())
    str_reg_auc = roc_auc_score((np.array(str_label_reg) * 1).flatten(), str_prob_reg.flatten())
    str_irg_auc = roc_auc_score((np.array(str_label_irg) * 1).flatten(), str_prob_irg.flatten())
    log.write('-' * 15 + '\n')
    log.write('str_asb_auc: {}\n'.format(str_asb_auc))
    log.write('str_reg_auc: {}\n'.format(str_reg_auc))
    log.write('str_irg_auc: {}\n'.format(str_irg_auc))

    pn_typ_auc = roc_auc_score((np.array(pn_label_typ) * 1).flatten(), pn_prob_typ.flatten())
    pn_asp_auc = roc_auc_score((np.array(pn_label_asp) * 1).flatten(), pn_prob_asp.flatten())
    pn_asb_auc = roc_auc_score((np.array(pn_label_asb) * 1).flatten(), pn_prob_asb.flatten())
    log.write('-' * 15 + '\n')
    log.write('pn_typ_auc: {}\n'.format(pn_typ_auc))
    log.write('pn_asp_auc: {}\n'.format(pn_asp_auc))
    log.write('pn_asb_auc: {}\n'.format(pn_asb_auc))

    log.close()

    return     avg_acc,[prob,pn_prob,str_prob,pig_prob,rs_prob,dag_prob,bwv_prob,vs_prob],[
    np.array(nevu_label),np.array(bcc_label),np.array(mel_label),np.array(misc_label),np.array(sk_label)],[
    nevu_prob ,bcc_prob , mel_prob ,misc_prob ,sk_prob],seven_point_feature_list, [gt,pn_gt,
                                                                                  str_gt,pig_gt,
                                                                                  rs_gt,dag_gt,
                                                                                  bwv_gt,vs_gt]
    
    
