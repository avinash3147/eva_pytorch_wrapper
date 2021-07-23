"""
    Contains all Methods related to data
"""
import torch, os, csv
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import datasets
from eva_pytorch_wrapper.utils.transforms_utility import *


def download_train_data(train_transforms, dataset_type):
    """
    Download Train Data
    Args:
        train_transforms: Apply transformations on train data
        dataset_type: MNIST or CIFAR10
    """
    data_set = datasets.CIFAR10 if dataset_type == 'CIFAR10' else datasets.MNIST
    train_data = data_set(
        './data',
        train= True,
        download= True,
        transform= train_transforms
    )
    return train_data

def download_test_data(test_transforms, dataset_type):
    """
    Download Test Data
    Args:
        test_traansforms: Apply transformations on test data
        dataset_type: MNIST or CIFAR10
    """
    data_set = datasets.CIFAR10 if dataset_type == 'CIFAR10' else datasets.MNIST
    test_data = data_set(
        './data',
        train= False,
        download= True,
        transform= test_transforms
    )
    return test_data

def load_train_data(train_data, **data_loader_args):
    """Load Train Data

    Args:
        train_data ([type]): [downloaded train data]
    """
    train_loader = torch.utils.data.DataLoader(
        train_data,
        **data_loader_args
    )
    return train_loader

def load_test_data(test_data, **data_loader_args):
    """Load Test Data

    Args:
        test_data ([type]): [Downloaded Test Data]
    """
    test_data = torch.utils.data.DataLoader(
        test_data,
        **data_loader_args
    )
    return test_data


def get_train_transformations(data_augmentation_type, mean=None, std=None, is_default_transforms=False):
    """Load Transformation Based on data augmentation type

    Args:
        data_augmentation_type ([String]): Class Names Present in transform utility file
        mean ([tuple]): Mean of the dataset
        std ([tuple]): Standard Deviation of the Dataset
        is_default_transforms boolean: whether to apply albumentation or not
    Returns:
        [type]: [description]
    """
    if is_default_transforms:
        return eval(data_augmentation_type)().default_transforms()
    return eval(data_augmentation_type)().train_transform(mean, std)


def get_test_transformations(data_augmentation_type, mean=None, std=None, is_default_transforms=False):
    """Load Transformation Based on data augmentation type

    Args:
        data_augmentation_type ([String]): Class Names Present in transform utility file
        mean ([tuple]): Mean of the dataset
        std ([tuple]): Standard Deviation of the Dataset
        is_default_transforms boolean: whether to apply albumentation or not

    Returns:
        [type]: [description]
    """
    if is_default_transforms:
        return eval(data_augmentation_type)().default_transforms()
    return eval(data_augmentation_type)().test_transform(mean, std)


def classes():
    id_dict = {}
    all_classes = {}
    for i, line in enumerate(open('./data/tiny-imagenet-200/wnids.txt', 'r')):
        id_dict[line.replace('\n', '')] = i

    result = {}
    class_id = {}
    for i, line in enumerate(open('./data/tiny-imagenet-200/words.txt', 'r')):
        n_id, word = line.split('\t')[:2]
        all_classes[n_id] = word
    for key, value in id_dict.items():
        result[value] = (all_classes[key].replace('\n', '').split(",")[0])
        class_id[key] = (value, all_classes[key])

    return result, class_id

def download():
    import requests, zipfile
    from io import BytesIO
    from tqdm.notebook import tqdm

    r = requests.get('http://cs231n.stanford.edu/tiny-imagenet-200.zip', stream=True)
    print('Downloading TinyImageNet Data')
    zip_ref = zipfile.ZipFile(BytesIO(r.content))
    for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
        zip_ref.extract(member=file, path='./data/')
    zip_ref.close()


class TinyImageNet(Dataset):
    def __init__(self, train=True, transform=None, train_split=0.7):
        self.image_paths = []
        self.targets = []
        self.transform = transform

        download()
        _, class_id = classes()

        # train images
        train_path = './data/tiny-imagenet-200/train'
        for class_dir in os.listdir(train_path):
            train_images_path = os.path.join(train_path, class_dir, 'images')
            for image in os.listdir(train_images_path):
                if image.endswith('.JPEG'):
                    self.image_paths.append(os.path.join(train_images_path, image))
                    self.targets.append(class_id[class_dir][0])
        self.indices = np.arange(len(self.targets))

        # val images
        val_path = './data/tiny-imagenet-200/val'
        val_images_path = os.path.join(val_path, 'images')
        with open(os.path.join(val_path, 'val_annotations.txt')) as f:
            for line in csv.reader(f, delimiter='\t'):
                self.image_paths.append(os.path.join(val_images_path, line[0]))
                self.targets.append(class_id[line[1]][0])

        self.indices = np.arange(len(self.targets))

        random_seed = 1
        np.random.seed(random_seed)
        np.random.shuffle(self.indices)

        split_idx = int(len(self.indices) * train_split)
        self.indices = self.indices[:split_idx] if train else self.indices[split_idx:]

    def __getitem__(self, index):

        image_index = self.indices[index]
        filepath = self.image_paths[image_index]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.targets[image_index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)
