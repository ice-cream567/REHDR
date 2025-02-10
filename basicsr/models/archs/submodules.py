import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.nn import init


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# class ConvLayere2vid(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
#         super(ConvLayer, self).__init__()
#
#         bias = False if norm == 'BN' else True
#         self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
#         if activation is not None:
#             self.activation = getattr(torch, activation, 'relu')
#         else:
#             self.activation = None
#
#         self.norm = norm
#         if norm == 'BN':
#             self.norm_layer = nn.BatchNorm2d(out_channels)
#         elif norm == 'IN':
#             self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)
#
#     def forward(self, x):
#         out = self.conv2d(x)
#
#         if self.norm in ['BN', 'IN']:
#             out = self.norm_layer(out)
#
#         if self.activation is not None:
#             out = self.activation(out)
#
#         return out


class ConvLayer(nn.Module):
    """
    conv norm relu

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, relu_slope=None, norm=None):
        super(ConvLayer, self).__init__()
        self.relu_slope = relu_slope

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if relu_slope is not None:
            if type(relu_slope) is str:
                self.relu = nn.ReLU()
            else:
                self.relu = nn.LeakyReLU(relu_slope, inplace=False)

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.relu_slope is not None:
            out = self.relu(out)

        return out


class TransposedConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None, norm=None):
        super(TransposedConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.transposed_conv2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=2, padding=padding, output_padding=1, bias=bias)
        self.conv_block = ResidualBlock(out_channels, out_channels)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.transposed_conv2d(x)

        out = self.conv_block(out)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(UpsampleConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        x_upsampled = f.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.conv2d(x_upsampled)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class RecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 recurrent_block_type='convlstm', activation='relu', norm=None):
        super(RecurrentConvLayer, self).__init__()

        assert (recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        # self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride, padding, activation, norm)
        self.conv = ImageEncoderConvBlock(in_channels, out_channels,
                                          downsample=True, relu_slope=0.2)
        self.recurrent_block = RecurrentBlock(input_size=out_channels, hidden_size=out_channels, kernel_size=3)
        self.fusion = Fusion(in_channels * 2)

    def forward(self, img, event, prev_state,head):

        img = self.conv(img)
        # fusion
        x = self.fusion(img, event,head)
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        return x, state


class Fusion(nn.Module):
    def __init__(self, out_size, use_bn=True, use_relu=True):
        super(Fusion, self).__init__()
        self.use_bn = use_bn
        self.use_relu = use_relu

        self.alise = nn.Conv2d(out_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.alise2 = nn.Conv2d(out_size * 2, out_size, kernel_size=3, stride=1, padding=1, bias=False)

        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_size)
            self.bn2 = nn.BatchNorm2d(out_size)

        if self.use_relu:
            self.relu = nn.ReLU(inplace=True)

        self.att = CALayer(out_size)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.out = nn.Conv2d(out_size, out_size, 1, 1, 0)
        self.sam = SAM(out_size)
    def forward(self, img_low, fea_high,head):
        mask_ldr =self.sam(img_low,head)
        img_low = mask_ldr*img_low
        x1 = self.alise2(torch.cat([img_low, fea_high], dim=1))

        if self.use_bn:
            x1 = self.bn1(x1)

        if self.use_relu:
            x1 = self.relu(x1)

        emerge = self.gap(x1)
        fea1 = self.alise(emerge)
        fea2 = self.alise(emerge)
        attention_vectors = torch.cat([fea1, fea2], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        low_att, high_att = torch.chunk(attention_vectors, 2, dim=1)
        high = fea_high * high_att
        low = img_low * low_att
        out1 = self.out(high + low)

        x2 = self.att(out1)
        x3 = self.alise(x2)

        if self.use_bn:
            x3 = self.bn2(x3)

        return x3


class DownsampleRecurrentConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, recurrent_block_type='convlstm', padding=0,
                 activation='relu'):
        super(DownsampleRecurrentConvLayer, self).__init__()

        self.activation = getattr(torch, activation, 'relu')

        assert (recurrent_block_type in ['convlstm', 'convgru'])
        self.recurrent_block_type = recurrent_block_type
        if self.recurrent_block_type == 'convlstm':
            RecurrentBlock = ConvLSTM
        else:
            RecurrentBlock = ConvGRU
        self.recurrent_block = RecurrentBlock(input_size=in_channels, hidden_size=out_channels, kernel_size=kernel_size)

    def forward(self, x, prev_state):
        state = self.recurrent_block(x, prev_state)
        x = state[0] if self.recurrent_block_type == 'convlstm' else state
        x = f.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return self.activation(x), state


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, norm=None):
        super(ResidualBlock, self).__init__()
        bias = False if norm == 'BN' else True
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.norm = norm
        if norm == 'BN':
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.bn1 = nn.InstanceNorm2d(out_channels)
            self.bn2 = nn.InstanceNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.norm in ['BN', 'IN']:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.norm in ['BN', 'IN']:
            out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class ConvLSTM(nn.Module):
    """Adapted from: https://github.com/Atcold/pytorch-CortexNet/blob/master/model/ConvLSTMCell.py """

    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        pad = kernel_size // 2

        # cache a tensor filled with zeros to avoid reallocating memory at each inference step if --no-recurrent is enabled
        self.zero_tensors = {}

        self.Gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=pad)

    def forward(self, input_, prev_state=None):

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:

            # create the zero tensor if it has not been created already
            state_size = tuple([batch_size, self.hidden_size] + list(spatial_size))
            if state_size not in self.zero_tensors:
                # allocate a tensor with size `spatial_size`, filled with zero (if it has not been allocated already)
                self.zero_tensors[state_size] = (
                    torch.zeros(state_size).to(input_.device),
                    torch.zeros(state_size).to(input_.device)
                )

            prev_state = self.zero_tensors[tuple(state_size)]

        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return [hidden, cell]


class ConvGRU(nn.Module):
    """
    Generate a convolutional GRU cell
    Adapted from: https://github.com/jacobkimmel/pytorch_convgru/blob/master/convgru.py
    """

    def __init__(self, input_size, hidden_size, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

        init.orthogonal_(self.reset_gate.weight)
        init.orthogonal_(self.update_gate.weight)
        init.orthogonal_(self.out_gate.weight)
        init.constant_(self.reset_gate.bias, 0.)
        init.constant_(self.update_gate.bias, 0.)
        init.constant_(self.out_gate.bias, 0.)

    def forward(self, input_, prev_state):
        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]

        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = torch.zeros(state_size).to(input_.device)

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update

        return new_state


############## image encoder #####################
class ImageEncoderConvBlock(nn.Module):
    """
    x conv relu conv relu +  conv_down(k=4 s=2 nobias)
    |------conv-----------|
    """

    def __init__(self, in_size, out_size, downsample, relu_slope):  # cat
        super(ImageEncoderConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        # self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=5, stride=1, padding=2, bias=True)
        # self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        # self.conv_2 = nn.Conv2d(out_size, out_size,kernel_size=5, stride=1, padding=2, bias=True)
        # self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample:
            self.down = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None, mask=None, event_filter=None, merge_before_downsample=True):
        out = self.conv_1(x)
        out_conv1 = self.relu_1(out)
        out_conv2 = self.relu_2(self.conv_2(out_conv1))
        out = out_conv2 + self.identity(x)
        if self.downsample:
            out = self.down(out)

        return out


## Supervised Attention Module
## https://github.com/swz30/MPRNet
class SAM(nn.Module):
    def __init__(self, out_size):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_size, 3, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(3, out_size, kernel_size=3, padding=1, bias=True)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 使用插值进行下采样
        x = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        return x