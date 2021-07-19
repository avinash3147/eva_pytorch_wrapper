"""

"""
import numpy as np
from numpy.core.fromnumeric import std
import torch
from torch import optim

from models.resnet import ResNet18
from models.custom_resnet import CustomResnet

from utils.graph_utility import two_variable_plot, plot_accuracy_loss_curves, plot_misclassified_images
from utils.misc import process_cnfig, get_data_loader_args, check_cuda_availability,\
    get_model_summary, lr_finder, load_optimizer, run_epochs, get_wrong_predictions
from utils.data_utility import download_train_data, download_test_data, load_train_data, \
    load_test_data, get_train_transformations, get_test_transformations


config = process_cnfig(file_name='/home/avinash/my_work/eva6/eva_pytorch_wrapper/config/config.yaml')

train_transforms = get_train_transformations(
    data_augmentation_type= config['data_augmentation']['type'],
    mean= config['mean']['value'],
    std= config['std']['value']
) # Get Train Transformations

test_transforms = get_test_transformations(
    data_augmentation_type= config['data_augmentation']['type'],
    mean= config['mean']['value'],
    std= config['std']['value']
) # Get Test Tranformations

train_data = download_train_data(
    dataset_type= config['data_set']['type'],
    train_transforms= train_transforms
) # Get Train Data

test_data = download_test_data(
    dataset_type= config['data_set']['type'],
    test_transforms= test_transforms
) # Get Test Data

cuda = check_cuda_availability() # Check if cuda is available or not

data_loader_args = get_data_loader_args(cuda=cuda) # Get Data Loader Arguments

train_loader = load_train_data(
    train_data= train_data,
    **data_loader_args
) # Load Train Data

test_loader = load_test_data(
    test_data= test_data,
    **data_loader_args
) # Load Test Data

device = torch.device("cuda" if cuda else "cpu")

model = eval(config['model']['type'])().to(device)

get_model_summary(
    model= model,
    input_size= config['input_size']['value']
) # Get Model Summary

lr_train_accuracy, test_lr = lr_finder(
    train_loader= train_loader,
    device= device,
    model= model,
    max_lr=config['max_lr']['value'],
    min_lr=config['min_lr']['value'],
    epochs=config['lr_finder_epochs']['value'],
    momentum=config['momentum']['value'],
    weight_decay=config['weight_decay']['value']
)

two_variable_plot(
    x= test_lr,
    y= lr_train_accuracy,
    xlabel= "Learning Rate",
    ylabel= "Train Accuracy",
    title= "LR v/s Accuracy"
) # Plot Accuracy vs LR

optimizer = load_optimizer(
    model= model,
    learning_rate= config['learning_rate']['value'],
    momentum=config['momentum']['value'], # Change to 0.005
    weight_decay=config['optimizer_weight_decay']['value']
) # load Optimizer

train_accuracy, train_losses, test_accuracy, test_losses, LR = run_epochs(
    train_loader= train_loader, 
    test_loader= test_loader, 
    device= device, 
    model= model, 
    optimizer= optimizer, 
    train_epochs= config['train_epochs']['value'], 
    pct_start= config['pct_start']['value'], 
    anneal_strategy= config['anneal_strategy']['value'],
    cycle_momentum= config['cycle_momentum']['value'],
    base_momentum= config['base_momentum']['value'],
    max_momentum= config['max_momentum']['value'],
    div_factor= config['div_factor']['value'],
    final_div_factor= config['final_div_factor']['value']
) # Run Train and Test Loop

two_variable_plot(
    x= np.arange(1, 25),
    y=LR,
    xlabel="Learning Rate",
    ylabel="Train Accuracy",
    title="LR vs Accuracy"
) # Plot LR vs Accuracy

plot_accuracy_loss_curves(
    train_accuracy= train_accuracy, 
    test_accuracy= test_accuracy, 
    train_losses= train_losses, 
    test_losses= test_losses
) # Plot Accuracy And Loss Curves

wrong_predictions, wrong_images, correct_label = get_wrong_predictions(
    model=model,
    test_loader= test_loader,
    device=device
) # Get Wrong Predictions

plot_misclassified_images(
    wrong_predictions= wrong_predictions,
    classes=config['classes']['value'],
    mean=config['mean']['value'],
    std=config['std']['value']
)

