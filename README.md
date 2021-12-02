# Introduction
Raw code of FusionM4Net: A multi-stage multi-modal learning algorithm for multi-label skin lesion classification.

# Dependencies
1. pytorch==1.8.0.
2. sklearn ==0.24.1.
3. opencv == 4.5.1.
4. numpy == 1.19.2.
5. keras == 2.4.3.
6. pandas == 1.2.4.
7. tqdm == 4.60.0.


# How to use it
1. Firstly, please download the Seven-Point Checklist dataset on http://derm.cs.sfu.ca.
2. Secondly, Please change the image path in dependency.py
3. Then, set data_mode = 'Normal' and data_mode = 'self_evaluated' to run FusionNet in main_cmv2.py
to get the corresponding weights respectively. 
4. Finally, run second_stage_fusion.ipynb sequently to get P1, P2, P3 respectively.
the Fusion scheme 1 is also in this ipynb file for convience.
Note that you need to change the image path "source_dir" according the dataset in your experiments.

Set data_mode = 'Normal' to run FusionNet is trained on the defaulted training and validation dataset to get 
the P_clin, P_derm and P_fusion, which are fused by Fusion Scheme 1 to obtain P_1

Set data_mode = 'self_evaluated' to run FusionNet is trained on our divided sub-training and sub-testing
to get the prediction information to train the SVM cluster in second stage.

More details, please see our paper "FusionM4Net: A multi-stage multi-modal learning algorithm for multi-label skin lesion classification" (DOI: https://doi.org/10.1016/j.media.2021.102307). 





