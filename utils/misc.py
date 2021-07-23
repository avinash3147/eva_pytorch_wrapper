import copy
from torch.cuda.random import manual_seed
import yaml

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm


def check_cuda_availability():
    """Check if cuda is available or not

    Returns:
        [Boolean]: True or False
    """
    seed = 1
    cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    
    if cuda:
        torch.cuda.manual_seed(seed)
    return cuda


def get_data_loader_args(cuda):
    """Load Data loader arguments based of cuda availability

    Args:
        cuda ([Boolean]): True or False
    """
    return dict(shuffle= True, batch_size=512, num_workers=2, pin_memory=True) if cuda else dict(shuffle= True, batch_size=64)


def process_cnfig(file_name):
    """Process configurtaion file

    Args:
        file_name ([path]): config file path

    Returns:
        [type]: configuration
    """
    with open(file_name, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            print("********** Loading configuration... **********")
            return config
        except ValueError:
            print("Invalid yaml file format.. Please provide good yaml file")
            exit(-1)
            
            
def get_model_summary(model, input_size):
    """Get Model Summary

    Args:
        model ([type]): Instance of Model Class
        input_size ([tuple]): conatins size of image and number of channels example: (3, 32, 32) for cifar10 image
    """
    from torchsummary import summary
    summary(model, input_size=eval(input_size))
    
    
def lr_finder(train_loader, device, model, max_lr, min_lr, epochs, momentum, weight_decay):
    """[summary]

    Args:
        train_loader ([type]): [description]
        device ([type]): [description]
        model ([type]): [description]
        max_lr ([type]): [description]
        min_lr ([type]): [description]
        epochs ([type]): [description]
        momentum ([type]): [description]
        weight_decay ([type]): [description]
    """
    Lrtest_train_acc = []
    LRtest_Lr = []

    criterion = nn.CrossEntropyLoss()

    step = (max_lr - min_lr) / epochs
    lr = min_lr
    for e in range(epochs):
        testmodel = copy.deepcopy(model)
        optimizer = optim.SGD(testmodel.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        lr += (max_lr - min_lr) / epochs
        testmodel.train()
        pbar = tqdm(train_loader)
        correct = 0
        processed = 0
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            y_pred = testmodel(data)
            loss = criterion(y_pred, target)
            loss.backward()
            optimizer.step()
            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            pbar.set_description(
                desc=f'epoch = {e + 1} Lr = {optimizer.param_groups[0]["lr"]}  Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        Lrtest_train_acc.append(100 * correct / processed)
        LRtest_Lr.append(optimizer.param_groups[0]['lr'])
    return Lrtest_train_acc, LRtest_Lr


def load_optimizer(model, learning_rate, momentum, weight_decay):
    """Load optimizer

    Args:
        model ([type]): [description]
        learning_rate ([type]): [description]
        momentum ([type]): [description]
        weight_decay ([type]): [description]
    """
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum= momentum, weight_decay= weight_decay)
    return optimizer


def run_epochs(train_loader, test_loader, device, model, optimizer, train_epochs, max_lr, pct_start=None,
               anneal_strategy=None, cycle_momentum=None, base_momentum=None, max_momentum=None, div_factor=None, final_div_factor=None):
    from eva_pytorch_wrapper.utils import train_utility, test_utility
    from torch.optim.lr_scheduler import OneCycleLR

    criterion = nn.CrossEntropyLoss()

    # if is_one_cycle_lr:
    LR = []
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=None, epochs=train_epochs,
                           steps_per_epoch=len(train_loader), pct_start=pct_start, anneal_strategy=anneal_strategy,
                           cycle_momentum=cycle_momentum, base_momentum=base_momentum, max_momentum=max_momentum, div_factor=div_factor,
                           final_div_factor=final_div_factor)
    # else:
    #     scheduler = OneCycleLR(optimizer, max_lr=max_lr, epochs=train_epochs, steps_per_epoch=len(train_loader))

    train_losses = []
    test_losses = []
    train_accuracy = []
    test_accuracy = []

    for epoch in range(1, train_epochs + 1):
        print("EPOCH:", epoch, 'LR:', optimizer.param_groups[0]['lr'])
        LR.append(optimizer.param_groups[0]['lr'])

        train_utility.train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_acc=train_accuracy,
            train_loss=train_losses
        )

        test_utility.test(
            model=model,
            device=device,
            test_loader=test_loader,
            test_acc=test_accuracy,
            test_losses=test_losses
        )

    return train_accuracy, train_losses, test_accuracy, test_losses, LR


def get_wrong_predictions(model, test_loader, device):
    """Get Wrong Predictions

    Args:
        model ([type]): [description]
        test_loader ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    wrong_images = []
    wrong_label = []
    correct_label = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()

            wrong_pred = (pred.eq(target.view_as(pred)) == False)
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])

            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
        print(f'Total wrong predictions are {len(wrong_predictions)}')

    return wrong_predictions, wrong_images, correct_label
    