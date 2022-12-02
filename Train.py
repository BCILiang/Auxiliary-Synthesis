import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from EarlyStopping import EarlyStopping
from Model import AssistClassifier, Generator6Layer
import torch.utils.data as da

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


def data_loader(data, label, batch=16, shuffle=True, drop=False):
    """
    Preprocess the data to fit model.
    Feed data into data_loader.
    input:
        data (float): samples*length*ch (samples*ch*length).
        label (int): samples, ie.: [0, 1, 1, 0, ..., 2].
        batch (int): batch size
        shuffle (bool): shuffle data before input into decoder
        drop (bool): drop the last samples if True
    output:
        data loader
    """
    label = torch.LongTensor(label.flatten()).to(device)
    if data.shape[1] >= data.shape[2]:
        data = torch.tensor(data.swapaxes(1, 2))
    data = torch.unsqueeze(data, dim=1).type('torch.FloatTensor').to(device)
    data = da.TensorDataset(data, label)
    loader = da.DataLoader(dataset=data, batch_size=batch, shuffle=shuffle, drop_last=drop)
    return loader


def train_decoder(train_x, train_y, test_x, test_y, ep=200, batch=16):
    """
    input:
        train_x, test_x (float): samples*length*ch (samples*ch*length).
        train_y, test_y (int): samples, ie.: [0, 1, 1, 0, ..., 2].
        ep (int): total train and test epoch
        batch (int): batch size
    output:
        train acc, test acc, weight_file
    """
    # Define training configuration
    assis = AssistClassifier(classes_num=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(assis.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5, cooldown=0, min_lr=0,
                                               verbose=True)
    metric = 'loss'
    early_stopping = EarlyStopping(10, metric=metric)

    # Define data loader
    train_loader = data_loader(train_x, train_y, batch=batch)
    test_loader = data_loader(test_x, test_y, batch=batch)

    train_acc = []
    test_acc = []
    for epoch in range(ep):
        # Train decoder
        assis.train()
        train_loss = 0
        correct = 0
        total = 0
        loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = assis(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(train_loader), 'Epoch: %d | AssisNet: trainLoss: %.4f | trainAcc: %.4f%% (%d/%d)'
                  % (epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        scheduler.step(loss)
        train_acc.append(round(correct / total, 4))

        # Test decoder
        assis.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                outputs = assis(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                print(batch_idx, len(test_loader), 'Epoch: %d | AssisNet: testLoss: %.4f | testAcc: %.4f%% (%d/%d)'
                      % (epoch, val_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        test_acc.append(round(correct / total, 4))

        early_stopping(val_loss if metric == 'loss' else test_acc[-1], assis)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Select the result before early stopping epoch.
    train_acc = np.asarray(train_acc[-11])
    test_acc = np.asarray(test_acc[-11])
    return train_acc, test_acc


def train_generator(data, label, ep=300, batch=16, alpha=1.0, beta=0.0001):
    """
    input:
        same as synthesis_samples() function
    output:
        train acc, test acc, weight_file
    """
    # Define training configuration
    g = Generator6Layer().to(device)
    a = AssistClassifier(classes_num=4)
    a.load_state_dict(torch.load('ModelParameter.pt'))
    a.to(device)
    cross = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    optimizer = torch.optim.Adam(g.parameters(), lr=0.0003, betas=(0.5, 0.999))
    scheduler_g = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=5, cooldown=0, min_lr=0, verbose=True
    )

    # Define data loader, it's important to set shuffle False
    train_loader = data_loader(data, label, batch=batch, shuffle=False)
    z = torch.randn(len(data), 127, 1, 1).to(device)

    a.eval()
    g.train()
    aug_set = []
    aug_lab = []
    for epoch in range(ep):
        # Train generator
        loss = 0
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            if batch_idx != len(train_loader) - 1:
                nz = z[batch_idx * batch:batch_idx * batch + batch]
                outputs_g = g(nz, targets)
            else:
                nz = z[batch_idx * batch:]
                outputs_g = g(nz, targets)
            outputs_a = a(outputs_g)
            loss = (alpha * mse(outputs_g, inputs) + beta * cross(outputs_a, targets)) / 2
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs_a.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            print(batch_idx, len(train_loader), 'Epoch: %d | SynNet: trainLoss: %.3f | trainAcc: %.3f%% (%d/%d)'
                  % (epoch, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            # Output samples synthesised from the last epoch
            if epoch == ep - 1:
                aug_set = np.append(aug_set, torch.squeeze(outputs_g).data.cpu().numpy())
                aug_lab = np.append(aug_lab, targets[:].data.cpu().numpy())

        scheduler_g.step(loss)

    aug_set = np.asarray(aug_set).reshape([-1, data.shape[2], data.shape[1]])
    aug_lab = np.asarray(aug_lab)

    return aug_set, aug_lab


def synthesis_samples(data, label, ratio=2, ep=300, batch=16, alpha=1.0, beta=0.0001):
    """
    Synthesis different ratio artificial samples
    input:
        data (float): samples*length*ch (samples*ch*length).
        label (int): samples, ie.: [0, 1, 1, 0, ..., 2].
        ratio (int): expand ratio
        z (float): samples*127*1*1.
        ep (int): total train and test epoch
        batch (int): batch size
        alpha (float): coefficient of MSE loss
        beta (float): coefficient of CE loss, Suggested range: 1e-5 - 1e-3
    output:
        train acc, test acc, weight_file
    """
    syn_data, syn_label = train_generator(data, label, ep=ep, batch=batch, alpha=alpha, beta=beta)
    while len(syn_label) < ratio * len(label):
        new_data, new_label = train_generator(data, label, ep=ep, batch=batch, alpha=alpha, beta=beta)
        syn_data = np.concatenate([syn_data, new_data])
        syn_label = np.concatenate([syn_label, new_label])
    return syn_data, syn_label
