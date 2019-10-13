import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from pretrainedmodels.models.torchvision_models import pretrained_settings
import functools
import torch.utils.model_zoo as model_zoo

class ResNetEncoder(ResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False
        del self.fc

    def forward(self, x):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)

        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)

        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)

resnet_encoders = {
    'resnet18': {
        'encoder': ResNetEncoder,
        'pretrained_settings': pretrained_settings['resnet18'],
        'out_shapes': (512, 256, 128, 64, 64),
        'params': {
            'block': BasicBlock,
            'layers': [2, 2, 2, 2],
        },
    },
}

encoders = {}
encoders.update(resnet_encoders)

def get_encoder(name, encoder_weights=None):
    Encoder = encoders[name]['encoder']
    encoder = Encoder(**encoders[name]['params'])
    encoder.out_shapes = encoders[name]['out_shapes']

    if encoder_weights is not None:
        settings = encoders[name]['pretrained_settings'][encoder_weights]
        encoder.load_state_dict(model_zoo.load_url(settings['url']))

    return encoder


class Model(nn.Module):

    def __init__(self):
        super().__init__()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def predict(self, x):
        if self.training:
            self.eval()
        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)
        return x

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=not (use_batchnorm)),
            nn.ReLU(inplace=True),
        ]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

class SCSEModule(nn.Module):
    def __init__(self, ch, re=16):
        super().__init__()
        self.cSE = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(ch, ch//re, 1),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(ch//re, ch, 1),
                                 nn.Sigmoid()
                                )
        self.sSE = nn.Sequential(nn.Conv2d(ch, ch, 1),
                                 nn.Sigmoid())
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x

class CenterBlock(DecoderBlock):
    def forward(self, x):
        return self.block(x)

class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            use_batchnorm=True,
            center=False,
            attention_type=None
    ):
        super().__init__()
        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None
        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.layer5 = DecoderBlock(in_channels[4], out_channels[4],
                                   use_batchnorm=use_batchnorm, attention_type=attention_type)
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))
        self.initialize()
    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels
    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.center:
            encoder_head = self.center(encoder_head)
        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)
        return x

class Unet(EncoderDecoder):
    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center=False,  # usefull for VGG models
            attention_type=None
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )
        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
            attention_type=attention_type
        )
        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)
