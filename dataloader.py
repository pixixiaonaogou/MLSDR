import pandas as pd
import cv2
from matplotlib import pyplot as plt
from torch.utils import data
from torch.utils.data import DataLoader
from dependency import *
import torch
import numpy as np
from utils import encode_label,encode_meta_label
from keras.utils import to_categorical
import albumentations
#Build the Pytorch dataloader
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    ShiftScaleRotate,
    RandomBrightnessContrast,
)

aug = Compose(
    [
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
        RandomRotate90(p=0.5),
        RandomBrightnessContrast(p=0.5),
        #RandomContrast(p=0.5),
        #RandomBrightness(p=0.5),
        #RandomGamma(p=0.5)

    ],
    p=0.5)


def load_image(path, shape):
    img = cv2.imread(path)
    img = cv2.resize(img, (shape[0], shape[1]))

    return img


class SkinDataset(data.Dataset):
    def __init__(self, image_dir, img_info, file_list, shape, is_test=False, num_class=1):
        self.is_test = is_test
        self.image_dir = image_dir
        self.img_info = img_info
        self.file_list = file_list
        self.shape = shape
        self.num_class = num_class
        self.total_img_info = img_info
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]
        sub_img_info = self.total_img_info[file_id:file_id + 1]
                
        # get the clincal image path
        clinic_img_path = sub_img_info['clinic']
        # get the dermoscopy image path
        dermoscopy_img_path = sub_img_info['derm']
        # load the clinical image
        clinic_img = load_image(self.image_dir + clinic_img_path[file_id], self.shape)
        # load the dermoscopy image
        dermoscopy_img = load_image(self.image_dir + dermoscopy_img_path[file_id], self.shape)

        # Encode the diagnositic label
        diagnosis_label = sub_img_info['diagnosis'][file_id]
        for index_label, label in enumerate(label_list):
            if diagnosis_label in label:
                diagnosis_index = index_label
                diagnosis_label_one_hot = to_categorical(diagnosis_index, num_label)
            else:
                continue

        if not self.is_test:
            augmented = aug(image=clinic_img, mask=dermoscopy_img)
            clinic_img = augmented['image']
            dermoscopy_img = augmented['mask']

        total_label = encode_label(sub_img_info, file_id)
        # print(total_label)
        clinic_img = torch.from_numpy(np.transpose(clinic_img, (2, 0, 1)).astype('float32') / 255)
        dermoscopy_img = torch.from_numpy(np.transpose(dermoscopy_img, (2, 0, 1)).astype('float32') / 255)
        meta_data, _, _ = encode_meta_label(sub_img_info, file_id)

        return clinic_img, dermoscopy_img,  meta_data, [total_label[0], total_label[1], total_label[2], total_label[3],
                                        total_label[4], total_label[5], total_label[6], total_label[7]]
    


def demo_test():
    #train,val,test dataset spliting
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    train_index_list_1 = train_index_list[0:206]
    train_index_list_2 = train_index_list[206:]

    df = pd.read_csv(img_info_path)

    #Img Information
    index_num = 7
    img_info = df[index_num:index_num+1]
    clinic_path = img_info['clinic']
    dermoscopy_path = img_info['derm']
    #source_dir = '../release_v0/release_v0/images/'
    clinic_img = cv2.imread(source_dir+clinic_path[index_num])
    dermoscopy_img = cv2.imread(source_dir+dermoscopy_path[index_num])

    plt.subplot(121)
    plt.imshow(dermoscopy_img)

    plt.subplot(122)
    plt.imshow(clinic_img)
    plt.show()

def demo_run():
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    df = pd.read_csv(img_info_path)

    shape = (224, 224)
    batch_size = 16
    num_workers = 0
    train_skindataset = SkinDataset(image_dir=source_dir,
                                    img_info=df,
                                    file_list=train_index_list,
                                    shape=shape, is_test=False)
                                    
    train_dataloader = DataLoader(
        dataset=train_skindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True)

    val_skindataset = SkinDataset(image_dir=source_dir,
                                  img_info=df,
                                  file_list=val_index_list,
                                  shape=shape, is_test=True)

    val_dataloader = DataLoader(
        dataset=val_skindataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True)

    for clinic_img,derm_img,label in train_dataloader:
        print(clinic_img.shape,derm_img.shape,label[0].shape)
        print('train_dataloader finished')

    for clinic_img,derm_img,label in val_dataloader:
        print(clinic_img.shape,derm_img.shape,label[0].shape)
        print('val_dataloader finished')

def generate_dataloader(shape,batch_size,num_workers,data_mode):
    test_index_df = pd.read_csv(test_index_path)
    train_index_df = pd.read_csv(train_index_path)
    val_index_df = pd.read_csv(val_index_path)

    train_index_list = list(train_index_df['indexes'])
    val_index_list = list(val_index_df['indexes'])
    test_index_list = list(test_index_df['indexes'])

    train_index_list_1 = train_index_list[0:206]
    train_index_list_2 = train_index_list[206:]


    df = pd.read_csv(img_info_path)
    if data_mode == 'self_evaluated':
      data_mode = 'SP'
      train_skindataset = SkinDataset(image_dir=source_dir,
                                      img_info=df,
                                      file_list=train_index_list_1,
                                      shape=shape, is_test=False,
                                      )
      train_dataloader = DataLoader(
          dataset=train_skindataset,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=True,
          shuffle=True)
  
      val_skindataset = SkinDataset(image_dir=source_dir,
                                    img_info=df,
                                    file_list=train_index_list_2,
                                    shape=shape, is_test=True,
                                    )
  
      val_dataloader = DataLoader(
          dataset=val_skindataset,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=True,
          shuffle=True)

    else:
      train_skindataset = SkinDataset(image_dir=source_dir,
                                      img_info=df,
                                      file_list=train_index_list,
                                      shape=shape, 
                                      is_test=False,
                                      )
      train_dataloader = DataLoader(
          dataset=train_skindataset,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=True,
          shuffle=True)
  
      val_skindataset = SkinDataset(image_dir=source_dir,
                                    img_info=df,
                                    file_list=val_index_list,
                                    shape=shape, 
                                    is_test=True,
                                    )
  
      val_dataloader = DataLoader(
          dataset=val_skindataset,
          batch_size=batch_size,
          num_workers=num_workers,
          pin_memory=True,
          shuffle=True)

    return train_dataloader, val_dataloader
