"""
Copyright 2020 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""
from functools import partial, reduce
import logging
import os
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)

INET_MEAN = [0.485, 0.456, 0.406]
INET_STD = [0.229, 0.224, 0.225]
INET_IMG_SIZE = 224

HDFS_ROOT = 'hdfs:/user/magicpony/visual_sentiment/devel/pytorch_model_zoo/imagenet/'
DEFAULT_CACHE_DIR = './pytorch_model_zoo_imagenet/'

NETWORK_REL_PATH = {
    'alexnet': 'alexnet',
    'densenet121': 'densenet/densenet121',
    'densenet161': 'densenet/densenet161',
    'densenet169': 'densenet/densenet169',
    'densenet201': 'densenet/densenet201',
    'inception_v3': 'inception/inception_v3',
    'resnet18': 'resnet/resnet18',
    'resnet34': 'resnet/resnet34',
    'resnet50': 'resnet/resnet50',
    'resnet101': 'resnet/resnet101',
    'resnet152': 'resnet/resnet152',
    'squeezenet1_0': 'squeezenet/squeezenet1_0',
    'squeezenet1_1': 'squeezenet/squeezenet1_1',
    'vgg11': 'vgg/vgg11',
    'vgg11_bn': 'vgg/vgg11_bn',
    'vgg13': 'vgg/vgg13',
    'vgg13_bn': 'vgg/vgg13_bn',
    'vgg16': 'vgg/vgg16',
    'vgg16_bn': 'vgg/vgg16_bn',
    'vgg19': 'vgg/vgg19',
    'vgg19_bn': 'vgg/vgg19_bn'
}

NETWORK_IDS = NETWORK_REL_PATH.keys()


def get_network(net_name, pretrained=True):
  if net_name not in NETWORK_IDS:
    raise ValueError('unknown network model {}'.format(net_name))

  # get model function
  net_fnc = getattr(models, net_name)

  return net_fnc(pretrained)

def switch_head(net_name, model, output_size):
  # replace network head with a fully connected layer with output_size of neurons
  if net_name.startswith('resnet') or net_name.startswith('inception'):  # fc
    n_fc_in = model.fc.in_features
    model.fc = nn.Linear(in_features=n_fc_in, out_features=output_size, bias=True)
  elif net_name.startswith('densenet'):  # classifier
    n_fc_in = model.classifier.in_features
    model.classifier = nn.Linear(in_features=n_fc_in, out_features=output_size, bias=True)
  elif net_name.startswith('vgg') or net_name.startswith('alexnet'):  # classifier/6
    n_fc_in = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features=n_fc_in, out_features=output_size, bias=True)
  else:
    raise ValueError('Network {} is not supported by switch_head'.format(net_name))

  return model


def set_all_layers(model, freeze=False):
  for param in model.parameters():
    param.requires_grad = not freeze


def set_layers_to_learn(model, layer_list):
  # freeze all layers
  set_all_layers(model, freeze=True)
  # unfreeze layers set to learn
  for layer in layer_list:
    ly = getattr(model, layer, None)
    if ly is None:
      named_layers = [k for k, _ in model.named_children()]
      raise ValueError(
          'layer name <{}> is not found in the model, possible choices {}'.format(
              layer, named_layers))

    pi = 0
    for l_param in ly.parameters():
      l_param.requires_grad = True
      pi += 1
    logger.debug('Layer {}: setting {} param tensor to train'.format(layer, pi))

  return model


def get_model_parameters(model, only_trainable=False):
  if only_trainable:
    return filter(lambda p: p.requires_grad, model.parameters())
  else:
    return model.parameters()


def model_info(model):
  info = 'Model: {}'.format(type(model))
  # get number of trainable params
  reqgr_t, reqgr_p = 0, 0
  nogr_t, nogr_p = 0, 0
  for param in model.parameters():
    if param.requires_grad:
      reqgr_t += 1
      reqgr_p += reduce((lambda x, y: x * y), list(param.size()))
    else:
      nogr_t += 1
      nogr_p += reduce((lambda x, y: x * y), list(param.size()))
  info += ', [{}/{} tensors, {}/{} parameters] are trainable'.format(
    reqgr_t, reqgr_t + nogr_t, reqgr_p, reqgr_p + nogr_p)
  return info


def setup_model(output_size, device, opt, load_state=False):
  logger.info('get network {}'.format(opt.net_name))
  model = get_network(opt.net_name, getattr(opt, 'net_pretrained', False))
  logger.info('switch network head')
  model = switch_head(opt.net_name, model, output_size=output_size)
  # logger.info(model)
  # load checkpoint
  if load_state and not getattr(opt, 'resume', None) is None:
    checkpoint = load_checkpoint(opt.resume)
    if 'model_state' in checkpoint.keys():
      model.load_state_dict(checkpoint['model_state'])
      logger.info('loaded model_state from checkpoint: {}'.format(opt.resume))
      # unfreeze all layers
      set_all_layers(model, freeze=False)
    else:
      logger.warn('did not find model_state in checkpoint! continue without')

  logger.info('using device: {}'.format(device))
  model = model.to(device)
  logger.info(model_info(model))
  # freeze/unfreeze layers
  if not getattr(opt, 'train_layers', None) is None and len(opt.train_layers) > 0:
    opt.train_layers = opt.train_layers.split(',')
    logger.info('setting only layers {} to train'.format(opt.train_layers))
    model = set_layers_to_learn(model, opt.train_layers)
    logger.info(model_info(model))

  return model


def add_model_parser_arguments(parser):
  parser.add_argument(
      '--net_name',
      type=str,
      default='resnet50',
      choices=NETWORK_IDS,
      help='name of neural network from model zoo')
  parser.add_argument(
      '--net_pretrained',
      action='store_true',
      help='load pretrained weights from imagenet')
  parser.add_argument('--resume', type=str, default=None, help='resume from this checkpoint file')
  parser.add_argument('--train_layers', type=str, default=None,
                      help='train only this list of layers, comma seperated list ')
  return parser


def get_optimizer(optim_name, param, opt):
  optim_fn = getattr(optim, optim_name, None)
  if optim_fn is None:
    raise ValueError('unknown optimization method <{}>'.format(optim_name))

  if optim_name == 'Adam':
    optim_fn = partial(optim_fn,
                       lr=getattr(opt, 'lr', 0.01),
                       betas=(getattr(opt, 'b1', 0.9), getattr(opt, 'b2', 0.999)),
                       eps=getattr(opt, 'eps', 1e-08),
                       weight_decay=getattr(opt, 'weight_decay', 0),
                       amsgrad=getattr(opt, 'amsgrad', False)
                       )
  elif optim_name == 'SGD':
    optim_fn = partial(optim_fn,
                       lr=getattr(opt, 'lr', 0.001),
                       momentum=getattr(opt, 'momentum', 0.9),
                       dampening=getattr(opt, 'dampening', 0),
                       weight_decay=getattr(opt, 'weight_decay', 0),
                       nesterov=getattr(opt, 'nesterov', False)
                       )
  elif optim_name == 'Adagrad':
    optim_fn = partial(optim_fn,
                       lr=getattr(opt, 'lr', 0.01),
                       lr_decay=getattr(opt, 'lr_decay', 0),
                       weight_decay=getattr(opt, 'weight_decay', 0),
                       initial_accumulator_value=getattr(opt, 'initial_accumulator_value', 0)
                       )
  elif optim_name == 'Adadelta':
    optim_fn = partial(optim_fn,
                       lr=getattr(opt, 'lr', 1.0),
                       rho=getattr(opt, 'rho', 0.9),
                       eps=getattr(opt, 'eps', 1e-06),
                       weight_decay=getattr(opt, 'weight_decay', 0)
                       )
  else:
    logger.warn('only learning rate option is supported for <{}>'.format(optim_name))
    optim_fn = partial(optim_fn, lr=getattr(opt, 'lr', 0.001))

  if param is not None:
    return optim_fn(param)
  else:
    logger.warning('no model parameters were provided, returning a patial optimizer function')
    return optim_fn


def setup_optimizer(model, opt, load_state=False):
  optimizer = get_optimizer(
      getattr(
          opt, 'optimizer', 'SGD'), get_model_parameters(
          model, only_trainable=True), opt)
  if load_state and not getattr(opt, 'resume', None) is None:
    checkpoint = load_checkpoint(opt.resume)
    if 'optimizer_state' in checkpoint.keys():
      optimizer.load_state_dict(checkpoint['optimizer_state'])
      logger.info('loaded optimizer_state from checkpoint: {}'.format(opt.resume))
    else:
      logger.warn('did not find optimizer_state in checkpoint! continue without')

  return optimizer


def add_optim_parser_arguments(parser):
  parser.add_argument('--optimizer', type=str, default='SGD', help='optimization method')
  parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
  parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
  parser.add_argument('--momentum', type=float, default=0.0, help='SGD: learning rate momentum')
  parser.add_argument('--nesterov', action='store_true', help='SGD: enables Nesterov momentum ')
  parser.add_argument('--b1', type=float, default=0.9, help='Adam: beta 1 ')
  parser.add_argument('--b2', type=float, default=0.999, help='Adam: beta 2')
  parser.add_argument(
      '--amsgrad',
      action='store_true',
      help='Adam: whether to use the AMSGrad variant')
  parser.add_argument('--lr_decay', type=float, default=0.0, help='Adagrad: learning rate decay')
  parser.add_argument(
      '--rho',
      type=float,
      default=0.9,
      help='Adadelta: coefficient used for computing a running average of squared gradients')
  return parser


def get_default_image_transform(
  augmentation=False, color_augmentation=False, image_size=INET_IMG_SIZE):
  if augmentation:
    if color_augmentation:
      image_transform = transforms.Compose([
          transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
          transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=INET_MEAN, std=INET_STD)
      ])
    else:
      image_transform = transforms.Compose([
          transforms.RandomResizedCrop(size=image_size, scale=(0.5, 1.0)),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize(mean=INET_MEAN, std=INET_STD)
      ])
  else:
    image_transform = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.CenterCrop(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=INET_MEAN, std=INET_STD)
    ])

  return image_transform


def save_checkpoint(state, filename='checkpoint.pth.tar', is_best=False):
  with open(filename, 'wb') as f:
    torch.save(state, f)

  if is_best:
    b_dir = os.path.dirname(filename)
    shutil.copyfile(filename, os.path.join(b_dir, 'model_best.pth.tar'))


def load_checkpoint(filename):
  with open(filename, 'rb') as f:
    state = torch.load(f)

  return state


def create_and_init_model(model_name, model_state, output_size):
  model = get_network(model_name, False)
  model = switch_head(model_name, model, output_size=output_size)
  model.load_state_dict(model_state)
  return model
