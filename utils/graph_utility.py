import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def two_variable_plot(x, y, xlabel, ylabel, title):
    """Two Variable Plot

    Args:
        x ([tuple]): [description]
        y ([tuple]): [description]
        xlabel ([String]): X-axis title
        ylabel ([String]): Y-axis Title
        title ([String]): Plot Title
    """
    print("********************************")
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    print("********************************")
    
    
def plot_accuracy_loss_curves(train_accuracy, test_accuracy, train_losses, test_losses):
    """Plot Train And Test Accuracy and loss curves

    Args:
        train_accuracy ([type]): [description]
        test_accuracy ([type]): [description]
        train_loss ([type]): [description]
        test_loss ([type]): [description]
    """
    sns.set(style="whitegrid")
    sns.set(font_scale=1)

    fig, axs = plt.subplots(2, 2, figsize=(25, 15))
    plt.rcParams["figure.figsize"] = (25, 6)

    axs[0, 0].set_title("Training Loss")
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].set_title("Test Accuracy")

    axs[0, 0].plot(train_losses, label="Training Loss")
    axs[0, 0].set_xlabel('epochs')
    axs[0, 0].set_ylabel('loss')

    axs[1, 0].plot(train_accuracy, label="Training Accuracy")
    axs[1, 0].set_xlabel('epochs')
    axs[1, 0].set_ylabel('accuracy')

    axs[0, 1].plot(test_losses, label="Validation Loss")
    axs[0, 1].set_xlabel('epochs')
    axs[0, 1].set_ylabel('loss')

    axs[1, 1].plot(test_accuracy, label="Validation Accuracy")
    axs[1, 1].set_xlabel('epochs')
    axs[1, 1].set_ylabel('accuracy')
    
def plot_misclassified_images(wrong_predictions, classes, mean, std):
    """Plot Misclassified Images

    Args:
        wrong_predictions ([type]): [description]
        classes ([type]): [description]
        mean ([type]): [description]
        std ([type]): [description]
    """
    fig = plt.figure(figsize=(10, 12))
    fig.tight_layout()
    
    mean = eval(mean)
    std = eval(std)
    # classes = eval(classes)

    for i, (img, pred, correct) in enumerate(wrong_predictions[:20]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j] * std[j]) + mean[j]

        img = np.transpose(img, (1, 2, 0))  # / 2 + 0.5
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis('off')
        ax.set_title(f'\nactual : {classes[target.item()]}\npredicted : {classes[pred.item()]}',
                     fontsize=10)
        ax.imshow(img)

    plt.show()
    

def imshow(img,c = "" ):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(10,10))
    plt.imshow(np.transpose(npimg, (1, 2, 0)),interpolation='none')
    plt.title(c)
    
    