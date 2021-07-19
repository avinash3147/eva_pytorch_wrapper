from tqdm import tqdm


def train(model, device, train_loader, optimizer, train_acc, train_loss, scheduler, criterion):
    """Train Model

    Args:
        model ([type]): [description]
        device ([type]): [description]
        train_loader ([type]): [description]
        optimizer ([type]): [description]
        train_acc ([type]): [description]
        train_loss ([type]): [description]
        scheduler ([type]): [description]
        criterion ([type]): [description]
    """
    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = criterion(y_pred, target)

        train_loss.append(loss.data.cpu().numpy().item())

        # Backpropagation
        loss.backward()

        optimizer.step()
        scheduler.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}')
        train_acc.append(100 * correct / processed)
