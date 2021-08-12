# image root
test_index_path = '../release_v0/release_v0/meta/test_indexes.csv'
train_index_path = '../release_v0/release_v0/meta/train_indexes.csv'
val_index_path = '../release_v0/release_v0/meta/valid_indexes.csv'
img_info_path = '../release_v0/release_v0/meta/meta.csv'
source_dir = '../release_v0/release_v0/images/'

#label_list
nevus_list = ['blue nevus','clark nevus','combined nevus','congenital nevus','dermal nevus','recurrent nevus','reed or spitz nevus']
basal_cell_carcinoma_list = ['basal cell carcinoma']
melanoma_list = ['melanoma','melanoma (in situ)','melanoma (less than 0.76 mm)','melanoma (0.76 to 1.5 mm)','melanoma (more than 1.5 mm)','melanoma metastasis']
miscellaneous_list = ['dermatofibroma','lentigo','melanosis','miscellaneous','vascular lesion']
SK_list = ['seborrheic keratosis']
label_list = [nevus_list,basal_cell_carcinoma_list,melanoma_list,miscellaneous_list,SK_list]

#seven-point
pigment_network_label_list = [['absent'],['typical'],['atypical']]#'typical:1,atypical:2
streaks_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
pigmentation_label_list = [['absent'],
                      ['diffuse regular','localized regular'],
                      ['localized irregular','diffuse irregular']]#regular:1, irregular:2
regression_structures_label_list = [['absent'],
                               ['blue areas','combinations','white areas']]# present:1
dots_and_globules_label_list = [['absent'],['regular'],['irregular']]#regular:1, irregular:2
blue_whitish_veil_label_list = [['absent'],['present']]#present:1
vascular_structures_label_list = [['absent'],
                             ['within regression','arborizing','comma','hairpin','wreath'],
                             ['linear irregular','dotted']]

#num of each label
num_label = len(label_list)
num_pigment_network_label = len(pigment_network_label_list)
num_streaks_label = len(streaks_label_list)
num_pigmentation_label = len(pigmentation_label_list)
num_regression_structures_label = len(regression_structures_label_list)
num_dots_and_globules_label = len(dots_and_globules_label_list)
num_blue_whitish_veil_label = len(blue_whitish_veil_label_list)
num_vascular_structures_label = len(vascular_structures_label_list)

#metadata information
level_of_diagnostic_difficulty_label_list = ['low','medium','high']
evaluation_list = ['flat','palpable','nodular']
location_list = ['back','lower limbs','abdomen','upper limbs','chest',
                  'head neck','acral','buttocks','genital areas']
sex_list = ['female','male']
management_list = ['excision','clinical follow up','no further examination']

num_level_of_diagnostic_difficulty_label_list = len(level_of_diagnostic_difficulty_label_list)
num_evaluation_list = len(evaluation_list)
num_location_list = len(location_list)
num_sex_list = len(sex_list)
num_management_list = len(management_list)

#class_list
class_list = [num_label,
num_pigment_network_label,
num_streaks_label,
num_pigmentation_label,
num_regression_structures_label,
num_dots_and_globules_label,
num_blue_whitish_veil_label,
num_vascular_structures_label]