"""
An example for the pipeline pf Auxiliary Synthesis Framework.
"""
import numpy as np
from Train import train_decoder, synthesis_samples
from sklearn.model_selection import train_test_split


def framework_pipeline():
    # Pretraining decoder with real samples
    pre_train_acc, pre_test_acc = train_decoder(train_x, train_y, test_x, test_y)
    print('Raw accuracy: Train: %.4f%% | Test: %.4f%%' % (pre_train_acc, pre_test_acc))

    # Getting artificial samples and concatenating with real samples
    aug_data, aug_label = synthesis_samples(train_x, train_y, ratio=2)
    print('Number of synthesised data: %d' % len(aug_data))
    train_x_aug = np.concatenate([train_x, aug_data])
    train_y_aug = np.concatenate([train_y, aug_label])

    # Retraining the decoder with all samples and testing
    train_acc, test_acc = train_decoder(train_x_aug, train_y_aug, test_x, test_y)
    print('Raw accuracy: Train: %.4f%% | Test: %.4f%% \n'
          'Enhanced accuracy: Train: %.4f%% | Test: %.4f%%' % (pre_train_acc, pre_test_acc, train_acc, test_acc))


# Getting real samples and normalization
datasetX = np.load('A01_data_All.npy')
datasetY = np.load('A01_label_All.npy')
train_x, test_x, train_y, test_y = train_test_split(datasetX, datasetY, test_size=0.15, shuffle=True, random_state=0)

# Enhancing
framework_pipeline()
