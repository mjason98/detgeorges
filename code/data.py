from torchvision import transforms, datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import os

from .utils import data_images_downloader

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets_d = {}
    datasets_d['train'] = Subset(dataset, train_idx)
    datasets_d['val'] = Subset(dataset, val_idx)
    return datasets_d

def save_data_indices(dataset, data_folder):
    with open(os.path.join(data_folder, 'train_index.txt'), 'w') as file:
        for key in dataset.class_to_idx:
            file.write(key + ' ' + str(dataset.class_to_idx[key]) + '\n')

def create_train_datasets(positive_csv:str, negative_csv:str, split=0.1, num_workers=4, image_size=224, data_folder='data'):
    '''
    Creates two datasets for training the model: 'train' the training set and 'val' the validation set during training.

    Parameters:

        positive_csv:str Path to the positive examples, where the first column of the CSV is the URL of the image.

        negative_csv:str Path to the negative examples, where the first column of the CSV is the URL of the image.

        split: Value in (0, 1). Percentage of the set destianed to the validation.

        image_size: int Image size after processing.

        data_folder: srt Directory where the images will be downloaded.
    
    Returns:

        (datasets, dataloaders) Two dictionaries with the datasets and dataloaders for 'train' and 'val'
    '''
    if not os.path.isdir("pts"):
        os.mkdir("pts")
    if not os.path.isdir("data"):
        os.mkdir("data")

    data_images_downloader(positive_csv, image_folder="pos", data_folder=data_folder)
    data_images_downloader(negative_csv, image_folder="neg", data_folder=data_folder)

    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # load all images
    image_dataset = datasets.ImageFolder(root='data/', transform=data_transform)
    save_data_indices(image_dataset, 'pts')

    #split the data in train dataset and val dataset
    datasets_d = train_val_dataset(image_dataset, split)
    dataloaders_d = {x:DataLoader(datasets_d[x],32, shuffle=True, num_workers=num_workers) for x in ['train','val']} 

    return datasets_d, dataloaders_d

