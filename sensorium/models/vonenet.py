from collections import OrderedDict
from torch import nn
from .modules import VOneBlock
from .back_ends import ResNetBackEnd, Bottleneck, AlexNetBackEnd, CORnetSBackEnd
from .params import generate_gabor_param
import numpy as np
from neuralpredictors.layers.cores.base import Core
import os
import torch
import warnings

def VOneNet(sf_corr=0.75, sf_max=9, sf_min=0, rand_param=False, gabor_seed=0,
            simple_channels=256, complex_channels=256,
            noise_mode='neuronal', noise_scale=0.35, noise_level=0.07, k_exc=25,
            model_arch='resnet50', image_size= (144, 256), visual_degrees=8, ksize=25, stride=4):


    out_channels = simple_channels + complex_channels

    sf, theta, phase, nx, ny = generate_gabor_param(out_channels, gabor_seed, rand_param, sf_corr, sf_max, sf_min)

    gabor_params = {'simple_channels': simple_channels, 'complex_channels': complex_channels, 'rand_param': rand_param,
                    'gabor_seed': gabor_seed, 'sf_max': sf_max, 'sf_corr': sf_corr, 'sf': sf.copy(),
                    'theta': theta.copy(), 'phase': phase.copy(), 'nx': nx.copy(), 'ny': ny.copy()}
    arch_params = {'k_exc': k_exc, 'arch': model_arch, 'ksize': ksize, 'stride': stride}


    # Conversions
    ppd = image_size[0] / visual_degrees

    sf = sf / ppd
    sigx = nx / sf
    sigy = ny / sf
    theta = theta/180 * np.pi
    phase = phase / 180 * np.pi

    vone_block = VOneBlock(sf=sf, theta=theta, sigx=sigx, sigy=sigy, phase=phase,
                           k_exc=k_exc, noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                           simple_channels=simple_channels, complex_channels=complex_channels,
                           ksize=ksize, stride=stride, input_size=image_size)

    '''
    if model_arch:
        bottleneck = nn.Conv2d(out_channels, 64, kernel_size=1, stride=1, bias=False)
        nn.init.kaiming_normal_(bottleneck.weight, mode='fan_out', nonlinearity='relu')

        if model_arch.lower() == 'resnet50':
            print('Model: ', 'VOneResnet50')
            model_back_end = ResNetBackEnd(block=Bottleneck, layers=[3, 4, 6, 3])
        elif model_arch.lower() == 'alexnet':
            print('Model: ', 'VOneAlexNet')
            model_back_end = AlexNetBackEnd()
        elif model_arch.lower() == 'cornets':
            print('Model: ', 'VOneCORnet-S')
            model_back_end = CORnetSBackEnd()

        model = nn.Sequential(OrderedDict([
            ('vone_block', vone_block),
            ('bottleneck', bottleneck),
            ('model', model_back_end),
        ]))
    else:
    ''';
    print('Model: ', 'VOneNet')
    model = vone_block

    model.image_size = image_size
    model.visual_degrees = visual_degrees
    model.gabor_params = gabor_params
    model.arch_params = arch_params

    return model
    

###
import requests
#from torch.nn import Module

FILE_WEIGHTS = {'alexnet': 'vonealexnet_e70.pth.tar', 'resnet50': 'voneresnet50_e70.pth.tar',
                'resnet50_at': 'voneresnet50_at_e96.pth.tar', 'cornets': 'vonecornets_e70.pth.tar',
                'resnet50_ns': 'voneresnet50_ns_e70.pth.tar'}


class Wrapper(nn.Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def get_model(model_arch='resnet50', pretrained=True, map_location='cpu', **kwargs):
    """
    Returns a VOneNet model.
    Select pretrained=True for returning one of the 3 pretrained models.
    model_arch: string with identifier to choose the architecture of the back-end (resnet50, cornets, alexnet)
    """
    if pretrained and model_arch:
        url = f'https://vonenet-models.s3.us-east-2.amazonaws.com/{FILE_WEIGHTS[model_arch.lower()]}'
        home_dir = os.environ['HOME']
        vonenet_dir = os.path.join(home_dir, '.vonenet')
        weightsdir_path = os.path.join(vonenet_dir, FILE_WEIGHTS[model_arch.lower()])
        if not os.path.exists(vonenet_dir):
            os.makedirs(vonenet_dir)
        if not os.path.exists(weightsdir_path):
            print('Downloading model weights to ', weightsdir_path)
            r = requests.get(url, allow_redirects=True)
            open(weightsdir_path, 'wb').write(r.content)

        ckpt_data = torch.load(weightsdir_path, map_location=map_location)

        stride = ckpt_data['flags']['stride']
        simple_channels = ckpt_data['flags']['simple_channels']
        complex_channels = ckpt_data['flags']['complex_channels']
        k_exc = ckpt_data['flags']['k_exc']

        noise_mode = ckpt_data['flags']['noise_mode']
        noise_scale = ckpt_data['flags']['noise_scale']
        noise_level = ckpt_data['flags']['noise_level']

        model_id = ckpt_data['flags']['arch'].replace('_','').lower()

        model = globals()[f'VOneNet'](model_arch=model_id, stride=stride, k_exc=k_exc,
                                      simple_channels=simple_channels, complex_channels=complex_channels,
                                      noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level)

        if model_arch.lower() == 'resnet50_at':
            ckpt_data['state_dict'].pop('vone_block.div_u.weight')
            ckpt_data['state_dict'].pop('vone_block.div_t.weight')
            model.load_state_dict(ckpt_data['state_dict'])
        else:
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module

        #model = nn.DataParallel(model)
    else:
        model = globals()[f'VOneNet'](model_arch=model_arch, **kwargs)
        #model = nn.DataParallel(model)

    model.to(map_location)
    return model


### Simone has been here
class TransferLearningCore(Core, nn.Module):
    """
    Core based on popular image recognition networks from torchvision such as VGG or AlexNet.
    Can be instantiated as random or pretrained. Core is frozen by default, which can be changed with the fine_tune
    argument.
    """

    def __init__(
        self,
        input_channels,
        #tl_model_name,
        layers,
        pretrained=True,
        final_batchnorm=True,
        final_nonlinearity=True,
        momentum=0.1,
        fine_tune=False,
        **kwargs,
    ):
        """
        Args:
            input_channels (int): Number of input channels. 1 if greyscale, 3 if RBG
            tl_model_name (str): Name of the image recognition Transfer Learning model. Possible are all models in
            torchvision, i.e. vgg16, alexnet, ...
            layers (int): Number of layers, i.e. after which layer to cut the original network
            pretrained (boolean): Whether to use a randomly initialized or pretrained network
            final_batchnorm (boolean): Whether to add a batch norm after the final conv layer
            final_nonlinearity (boolean): Whether to add a final nonlinearity (ReLU)
            momentum (float): Momentum term for batch norm. Irrelevant if batch_norm=False
            fine_tune (boolean): Whether to clip gradients before this core or to allow training on the core
            **kwargs:
        """
        if kwargs:
            warnings.warn(
                "Ignoring input {} when creating {}".format(repr(kwargs), self.__class__.__name__),
                UserWarning,
            )
        super().__init__()

        self.input_channels = input_channels
        self.momentum = momentum

        # Download model and cut after specified layer
        TL_model = get_model()  #getattr(torchvision.models, tl_model_name)(pretrained=pretrained)
        #print(TL_model)
        #print(list(TL_model.children()))
        TL_model_clipped = nn.Sequential(*list(TL_model.children())[:layers])
        #TL_model_clipped = nn.Sequential(*list(TL_model.features.children())[:layers])
        if not isinstance(TL_model_clipped[-1], nn.Conv2d):
            warnings.warn(
                "Final layer is of type {}, not nn.Conv2d".format(type(TL_model_clipped[-1])),
                UserWarning,
            )

        # Fix pretrained parameters during training
        if not fine_tune:
            for param in TL_model_clipped.parameters():
                param.requires_grad = False

        # Stack model together
        self.features = nn.Sequential()
        self.features.add_module("TransferLearning", TL_model_clipped)
        if final_batchnorm:
            self.features.add_module("OutBatchNorm", nn.BatchNorm2d(self.outchannels, momentum=self.momentum))
        if final_nonlinearity:
            self.features.add_module("OutNonlin", nn.ReLU(inplace=True))

    def forward(self, input_):
        # If model is designed for RBG input but input is greyscale, repeat the same input 3 times
        if self.input_channels == 1 and self.features.TransferLearning[0].in_channels == 3:
            input_ = input_.repeat(1, 3, 1, 1)
        input_ = self.features(input_)
        return input_

    def regularizer(self):
        return 0

    @property
    def outchannels(self):
        """
        Function which returns the number of channels in the output conv layer. If the output layer is not a conv
        layer, the last conv layer in the network is used.
        Returns: Number of output channels
        """
        found_outchannels = False
        i = 1
        while not found_outchannels:
            if "out_channels" in self.features.TransferLearning[-i].__dict__:
                found_outchannels = True
            else:
                i += 1
        return self.features.TransferLearning[-i].out_channels

    def initialize(self):
        logger.warning(
            "Ignoring initialization since the parameters should be acquired from a pretrained model. If you want random weights, set pretrained = False."
        )
