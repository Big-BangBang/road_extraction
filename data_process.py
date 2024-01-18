import os, cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import torch
from torch.utils.data import DataLoader
import albumentations as album

def get_dataloader():
    train_df, valid_df = get_train_dataframe()
    select_class_rgb_values = get_class_dict()

    train_dataset = RoadsDataset(train_df, augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(),class_rgb_values=select_class_rgb_values)
    valid_dataset = RoadsDataset(valid_df,preprocessing=get_preprocessing(), class_rgb_values=select_class_rgb_values)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader =  DataLoader(valid_dataset, batch_size=4, shuffle=True)
    return train_loader, valid_loader, len(train_loader) * 4, len(valid_loader) * 4
def get_train_dataframe():
    DATA_DIR = '/home/ywn/road_extraction_best/dataset'

    metadata_df = pd.read_csv(os.path.join(DATA_DIR, 'metadata.csv'))
    # print(metadata_df.head())

    metadata_df_train = metadata_df[metadata_df['split']=='train']
    metadata_df_train = metadata_df_train[['image_id', 'sat_image_path', 'mask_path']]
    metadata_df_train['sat_image_path'] = metadata_df_train['sat_image_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df_train['mask_path'] = metadata_df_train['mask_path'].apply(lambda img_pth: os.path.join(DATA_DIR, img_pth))
    metadata_df_train.head()

    metadata_df_train = metadata_df_train.sample(frac=1).reset_index(drop=True)

    valid_df = metadata_df_train.sample(frac=0.1, random_state=42)
    train_df = metadata_df_train.drop(valid_df.index)
    return train_df, valid_df




def get_class_dict():
    DATA_DIR = '/home/ywn/road_extraction_best/dataset'
    class_dict = pd.read_csv(os.path.join(DATA_DIR, 'class_dict.csv'))

    class_names = class_dict['name'].tolist()
    class_rgb_values = class_dict[['r','g','b']].values.tolist()

    # print('All dataset classes and their corresponding RGB values in labels:')
    # print('Class Names: ', class_names)
    # print('Class RGB values: ', class_rgb_values)


    select_classes = ['background', 'road']

    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    # print('Selected classes and their corresponding RGB values in labels:')
    # print('Class Names: ', class_names)
    # print('Class RGB values: ', class_rgb_values)
    return select_class_rgb_values



def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()
def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
def reverse_one_hot(image):
    x = np.argmax(image, axis = -1)
    return x
def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


class RoadsDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            df,
            class_rgb_values=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.image_paths = df['sat_image_path'].tolist()[0:500]
        self.mask_paths = df['mask_path'].tolist()[0:500]
        
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(self.mask_paths[i]), cv2.COLOR_BGR2RGB)
        
        mask = one_hot_encode(mask, self.class_rgb_values).astype('float')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.image_paths)
    
# dataset = RoadsDataset(train_df, class_rgb_values=select_class_rgb_values)
# random_idx = random.randint(0, len(dataset)-1)
# image, mask = dataset[random_idx]

# visualize(
#     original_image = image,
#     ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
#     one_hot_encoded_mask = reverse_one_hot(mask)
# )


def get_training_augmentation():
    train_transform = [
        album.HorizontalFlip(p=0.5),
        album.VerticalFlip(p=0.5),
    ]
    return album.Compose(train_transform)
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn=None):
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)
# train_loader, valid_loader, len_train, len_valid = get_dataloader()
# print(len_train)
# for batch in train_loader:
#     print(batch[0].shape, batch[1][0, 0:2, 0, 0])
#     break
# select_class_rgb_values = get_class_dict()
# train_df, _ = get_train_dataframe()
# dataset = RoadsDataset(train_df, class_rgb_values=select_class_rgb_values)
# random_idx = random.randint(0, len(dataset)-1)
# image, mask = dataset[random_idx]
# print(image.shape, mask.shape, type(image))