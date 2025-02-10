from basicsr.models.archs.base_model import BaseModel
import torch.nn as nn
import torch
from basicsr.models.archs.unet import UNet, UNetRecurrent
from os.path import join
from basicsr.models.archs.submodules import ImageEncoderConvBlock, ConvLSTM, ConvGRU, ResidualBlock, ConvLayer, \
    UpsampleConvLayer, \
    TransposedConvLayer, RecurrentConvLayer, SAM, DownsampleBlock


def skip_concat(x1, x2):
    return torch.cat([x1, x2], dim=1)


def skip_sum(x1, x2):
    return x1 + x2


class BaseUNet(nn.Module):
    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum', activation='sigmoid',
                 num_encoders=4, base_num_channels=32, num_residual_blocks=2, norm=None, use_upsample_conv=True):
        super(BaseUNet, self).__init__()

        self.ev_chn = ev_chn
        self.img_chn = img_chn
        self.out_chn = out_chn
        self.skip_type = skip_type
        self.apply_skip_connection = skip_sum if self.skip_type == 'sum' else skip_concat
        self.activation = activation
        self.norm = norm

        if use_upsample_conv:
            print('Using UpsampleConvLayer (slow, but no checkerboard artefacts)')
            self.UpsampleLayer = UpsampleConvLayer
        else:
            print('Using TransposedConvLayer (fast, with checkerboard artefacts)')
            self.UpsampleLayer = TransposedConvLayer

        self.num_encoders = num_encoders
        self.base_num_channels = base_num_channels
        self.num_residual_blocks = num_residual_blocks
        self.max_num_channels = self.base_num_channels * pow(2, self.num_encoders)

        assert (self.ev_chn > 0)
        assert (self.img_chn > 0)
        assert (self.out_chn > 0)

        self.encoder_input_sizes = []
        for i in range(self.num_encoders):
            self.encoder_input_sizes.append(self.base_num_channels * pow(2, i))

        self.encoder_output_sizes = [self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]

        self.activation = getattr(torch, self.activation, 'sigmoid')

    def build_resblocks(self):
        self.resblocks = nn.ModuleList()
        for i in range(self.num_residual_blocks):
            self.resblocks.append(ResidualBlock(self.max_num_channels, self.max_num_channels, norm=self.norm))

    def build_decoders(self):
        decoder_input_sizes = list(reversed([self.base_num_channels * pow(2, i + 1) for i in range(self.num_encoders)]))

        self.decoders = nn.ModuleList()
        for input_size in decoder_input_sizes:
            self.decoders.append(self.UpsampleLayer(input_size if self.skip_type == 'sum' else 2 * input_size,
                                                    input_size // 2,
                                                    kernel_size=5, padding=2, norm=self.norm))

    def build_prediction_layer(self):
        self.pred = ConvLayer(self.base_num_channels if self.skip_type == 'sum' else 2 * self.base_num_channels,
                              self.out_chn, 3, norm=self.norm)


class REHDR(BaseUNet):
    """
    Recurrent UNet architecture where every encoder is followed by a recurrent convolutional block,
    such as a ConvLSTM or a ConvGRU.
    Symmetric, skip connections on every encoding layer.
    """

    def __init__(self, img_chn, ev_chn, out_chn=3, skip_type='sum',
                 recurrent_block_type='convlstm', activation='sigmoid', num_encoders=3, base_num_channels=32,
                 num_residual_blocks=2, norm='BN', use_upsample_conv=False):
        super(REHDR, self).__init__(img_chn, ev_chn, out_chn, skip_type, activation,
                                    num_encoders, base_num_channels, num_residual_blocks, norm,
                                    use_upsample_conv)

        # self.head_ev = ConvLayer(self.ev_chn, self.base_num_channels,
        #                       kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W
        # self.head_img = ConvLayer(self.img_chn, self.base_num_channels,
        #                           kernel_size=5, stride=1, padding=2, relu_slope=0.2)  # N x C x H x W -> N x 32 x H x W

        self.head_ev = ConvLayer(self.ev_chn, self.base_num_channels,
                                 kernel_size=3, stride=1, padding=1,
                                 norm=self.norm)  # N x C x H x W -> N x 32 x H x W
        self.head_img = ConvLayer(self.img_chn, self.base_num_channels,
                                  kernel_size=3, stride=1, padding=1,
                                  norm=self.norm)  # N x C x H x W -> N x 32 x H x W
        self.ev_encoders = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.ev_encoders.append(ImageEncoderConvBlock(in_size=input_size, out_size=output_size,
                                                          downsample=True, relu_slope=0.2))

        self.img_encoders = nn.ModuleList()

        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            self.img_encoders.append(RecurrentConvLayer(input_size, output_size,
                                                        kernel_size=5, stride=2, padding=2,
                                                        recurrent_block_type=recurrent_block_type,
                                                        norm=self.norm))
        self.multiscale_layers = nn.ModuleList()
        for input_size, output_size in zip(self.encoder_input_sizes, self.encoder_output_sizes):
            layer = DownsampleBlock(input_size, output_size)
            self.multiscale_layers.append(layer)

        self.build_resblocks()
        self.build_decoders()
        self.build_prediction_layer()

    def forward(self, x, event, prev_states):
        """
        :param x: N x img_chn x H x W
        :param event: N x ev_chn x H x W
        :param prev_states: previous LSTM states for every encoder layer
        :return: N x out_chn x H x W
        """

        # head
        # x = torch.pow(x, 2.2)
        head0=x
        x = self.head_img(x)
        ev = self.head_ev(event)
        head = x

        # 创建多尺度特征图
        img = []
        current_tensor = head0
        for layer in self.multiscale_layers:
            current_tensor = layer(current_tensor)
            img.append(current_tensor)

        if prev_states is None:
            prev_states = [None] * self.num_encoders

        # evEncoder
        e_blocks = []
        for i, ev_encoder in enumerate(self.ev_encoders):
            ev = ev_encoder(ev)
            e_blocks.append(ev)

        # imgEncoder
        blocks = []
        states = []
        for i, encoder in enumerate(self.img_encoders):
            x, state = encoder(x, e_blocks[i], prev_states[i],img[i])
            blocks.append(x)
            states.append(state)

        # residual blocks
        for resblock in self.resblocks:
            x = resblock(x)

        # decoder
        for i, decoder in enumerate(self.decoders):
            x = decoder(self.apply_skip_connection(x, blocks[self.num_encoders - i - 1]))

        # tail
        img = self.activation(self.pred(self.apply_skip_connection(x, head)))
        # img = self.pred(self.apply_skip_connection(x, head))

        return img, states


if __name__ == '__main__':
    import time

    model = REHDR(img_chn=3, ev_chn=5)
    device = 'cuda'
    x = torch.rand(1, 3, 256, 256).to(device)
    event = torch.rand(1, 5, 256, 256).to(device)
    model = model.to(device)

    start_time = time.time()
    result = model(x, event, prev_states=None)
    end_time = time.time()

    inference_time = end_time - start_time
    print('Inference time:{}'.format(inference_time))
