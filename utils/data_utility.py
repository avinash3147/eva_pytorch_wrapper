"""
    Contains all Methods related to data
"""
import torch

from torchvision import datasets
from utils.transforms_utility import *


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

def get_train_transformations(data_augmentation_type, mean, std):
    """Load Transformation Based on data augmentation type

    Args:
        data_augmentation_type ([String]): Class Names Present in transform utility file
        mean ([tuple]): Mean of the dataset
        std ([tuple]): Standard Deviation of the Dataset

    Returns:
        [type]: [description]
    """
    return eval(data_augmentation_type)().train_transform(mean, std)

def get_test_transformations(data_augmentation_type, mean, std):
    """Load Transformation Based on data augmentation type

    Args:
        data_augmentation_type ([String]): Class Names Present in transform utility file
        mean ([tuple]): Mean of the dataset
        std ([tuple]): Standard Deviation of the Dataset

    Returns:
        [type]: [description]
    """
    return eval(data_augmentation_type)().test_transform(mean, std)