"""
Define Structure for:
Generator, Assist-Classifier,
EEGNet, DeepConvNet, ShallowConvNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as f


class Generator6Layer(nn.Module):
    def __init__(self):
        super(Generator6Layer, self).__init__()
        self.dropout = 0
        self.label_emb = nn.Embedding(4, 1)

        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=128,
                kernel_size=(4, 256),
                stride=(1, 1),
                padding=(0, 64),
                bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ELU(),
            # nn.Dropout(self.dropout)
        )

        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=(4, 128),
                stride=(2, 2),
                padding=(0, 64),
                bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ELU(),
            # nn.Dropout(self.dropout)
        )

        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=(3, 64),
                stride=(2, 2),
                padding=(0, 32),
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            # nn.Dropout(self.dropout)
        )

        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=16,
                kernel_size=(3, 32),
                stride=(1, 2),
                padding=(1, 16),
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.Dropout(self.dropout)
        )

        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16,
                out_channels=8,
                kernel_size=(3, 16),
                stride=(1, 1),
                padding=(1, 16),
                bias=False
            ),
            nn.BatchNorm2d(8),
            nn.ELU(),
            # nn.Dropout(self.dropout)
        )

        self.layer6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=8,
                out_channels=1,
                kernel_size=(2, 8),
                stride=(1, 1),
                padding=(0, 0),
                bias=False
            ),
        )

    def forward(self, z, lab, num, last=False):
        lab = self.label_emb(lab)
        bs = num if last else 16
        lab = lab.view(bs, 1, 1, 1)
        out = torch.cat((lab, z), dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        return out


class SEBlockLiner(nn.Module):
    """github url: , paper DOI: """

    def __init__(self, channel, reduction=16):
        super(SEBlockLiner, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class AssistClassifier(nn.Module):
    def __init__(self, classes_num=4):
        super(AssistClassifier, self).__init__()
        self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=16,
                kernel_size=(22, 1),
                bias=True,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Dropout(self.drop_out)
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 64),
                groups=16,
                bias=True
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(1, 1),
                bias=True
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)
        )

        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(1, 32),
                groups=32,
                bias=True
            ),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.attention = SEBlockLiner(32, reduction=16)

        self.out = nn.Linear((32 * 25), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class ConstrainedConv2d(nn.Conv2d):
    def forward(self, inp):
        return f.conv2d(inp, self.weight.clamp(min=-1.0, max=1.0), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConstrainedLiner(nn.Linear):
    def forward(self, inp):
        return f.linear(inp, self.weight.clamp(min=-0.25, max=0.25), self.bias)


class EEGNet(nn.Module):
    """
    pytorch github url: https://github.com/aliasvishnu/EEGNet
    tensorflow github url: https://github.com/vlawhern/arl-eegmodels paper
    DOI: 10.1088/1741-2552/aace8c
    """

    def __init__(self, classes_num):
        super(EEGNet, self).__init__()
        self.drop_out = 0.5

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((64, 63, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 128),  # filter size
                bias=False,
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        self.block_2 = nn.Sequential(
            ConstrainedConv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(22, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((16, 15, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 32),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 16)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = ConstrainedLiner((16 * 7), classes_num)  # 6 for zero padding

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class ShallowConvNet(nn.Module):
    """
    github url: https://github.com/robintibor/braindecode
    paper DOI:10. 1002/hbm.23730
    """

    def __init__(self, classes_num):
        super(ShallowConvNet, self).__init__()

        self.block_1 = nn.Sequential(
            # conv_time
            nn.Conv2d(
                in_channels=1,
                out_channels=40,
                kernel_size=(25, 1),
                stride=(1, 1),
            ),
            # conv_spat
            nn.Conv2d(
                in_channels=40,
                out_channels=40,
                kernel_size=(1, 22),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(40, momentum=0.1, affine=True),
        )

        self.block_pool = nn.Sequential(
            nn.AvgPool2d(
                kernel_size=(75, 1),
                stride=(15, 1),
            ),
        )

        self.block_drop = nn.Dropout(0.5)

        self.out = nn.Linear((40 * 61), classes_num)

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.block_1(x)
        x = x * x
        x = self.block_pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.block_drop(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x


class DeepConvNet(nn.Module):
    """
    github url: https://github.com/robintibor/braindecode
    paper DOI:10. 1002/hbm.23730
    """

    def __init__(self, classes_num):
        super(DeepConvNet, self).__init__()
        self.drop = 0.5

        self.block_1 = nn.Sequential(
            # conv_time
            nn.Conv2d(
                in_channels=1,
                out_channels=25,
                kernel_size=(10, 1),
                stride=(1, 1),
            ),
            # conv_spat
            nn.Conv2d(
                in_channels=25,
                out_channels=25,
                kernel_size=(1, 22),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(25, momentum=0.1, affine=True),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),
            nn.Dropout(self.drop)
        )

        self.block_2 = nn.Sequential(
            # conv_time
            nn.Conv2d(
                in_channels=25,
                out_channels=50,
                kernel_size=(10, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(50, momentum=0.1, affine=True),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),
            nn.Dropout(self.drop)
        )

        self.block_3 = nn.Sequential(
            # conv_time
            nn.Conv2d(
                in_channels=50,
                out_channels=100,
                kernel_size=(10, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(100, momentum=0.1, affine=True),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),
            nn.Dropout(self.drop)
        )

        self.block_4 = nn.Sequential(
            # conv_time
            nn.Conv2d(
                in_channels=100,
                out_channels=200,
                kernel_size=(10, 1),
                stride=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(200, momentum=0.1, affine=True),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),
            nn.Dropout(self.drop)
        )

        self.out = nn.Linear((200 * 7), classes_num)

    def forward(self, x):
        x = x.transpose(2, 3)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        return x
