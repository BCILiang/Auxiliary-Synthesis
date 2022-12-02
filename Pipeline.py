"""
An example for the pipeline pf Auxiliary Synthesis Framework.
"""
import numpy as np
from Train import train_decoder, synthesis_samples
from sklearn.model_selection import train_test_split


def framework_pipeline(train_x, train_y, test_x, test_y, s_hold, ratio=2, epoch=200, batch=16, alpha=1, beta=0.0001):
    """
        Synthesis different ratio artificial samples
    input:
        train_x, test_x (float): Train and test data, shape as: samples*length*ch (samples*ch*length).
        train_y, test_y (int): Train and test label, shape as: samples, ie.: [0, 1, 1, 0, ..., 2].
        s_hold (float):  It is suggested to determine by cross-validation on limited real data, an approximate value.
        ratio (int): Expand ratio
        epoch (int): Total train and test epoch
        batch (int): Batch size
        alpha (float): Coefficient of MSE loss, Suggested larger than beta
        beta (float): Coefficient of CE loss, Suggested range: 1e-5 - 1e-3
    output:
        None
    """
    # Pretraining decoder with real samples
    pre_test_acc, pre_train_acc = 0, 0
    while pre_test_acc <= s_hold:
        pre_train_acc, pre_test_acc = train_decoder(train_x, train_y, test_x, test_y, ep=epoch, batch=batch)
        print('Raw accuracy: Train: %.4f%% | Test: %.4f%%' % (pre_train_acc, pre_test_acc))

    # Getting artificial samples and concatenating with real samples
    aug_data, aug_label = synthesis_samples(train_x, train_y, ratio=ratio, ep=epoch, batch=batch, alpha=alpha, beta=beta)
    print('Number of synthesised data: %d' % len(aug_data))
    train_x_aug = np.concatenate([train_x, aug_data.swapaxes(1, 2)])
    train_y_aug = np.concatenate([train_y, aug_label])

    # Retraining the decoder with all samples and testing
    train_acc, test_acc = train_decoder(train_x_aug, train_y_aug, test_x, test_y)
    print('Raw accuracy: Train: %.4f%% | Test: %.4f%% \n'
          'Enhanced accuracy: Train: %.4f%% | Test: %.4f%%' % (pre_train_acc, pre_test_acc, train_acc, test_acc))


# Getting real samples and normalization
datasetX = np.load('A01_data_All.npy')
datasetY = np.load('A01_label_All.npy')
train_data, test_data, train_label, test_label = train_test_split(datasetX, datasetY, test_size=0.15, shuffle=True, random_state=0)

# Enhancing
framework_pipeline(train_data, train_label, test_data, test_label, 0.65)
