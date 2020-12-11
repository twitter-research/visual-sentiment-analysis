from __future__ import division

import argparse
from datetime import datetime
from functools import partial
import logging
import os
import shutil
import random

import data.datasets as datasets
from data.samplers import (collate_ignore_nones, PersistentSampler, WeightedClassRandomSampler)
from data.transformers import MultiLabelBinarizer
from model import nn_factory as nf
from model import metrics
from model.train import Trainer
from utils import log as log_util

import torch
from torch.utils.data.sampler import RandomSampler, SequentialSampler
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # data
  parser.add_argument('--train_csv', type=str, default=None, help='csv file with training samples')
  parser.add_argument('--val_csv', type=str, default=None, help='csv file with validation samples')
  parser.add_argument('--category_csv', default=None, type=str, help='csv file with category names')
  parser.add_argument('--save_dir', default=None, type=str, help='save directory')
  # model
  parser = nf.add_model_parser_arguments(parser)
  # optim
  parser = nf.add_optim_parser_arguments(parser)
  # scheduler
  parser.add_argument('--scheduler_step_size', type=int, default=-1,
                      help='period in epochs of learning rate decay')
  parser.add_argument('--scheduler_gamma', type=float, default=0.1,
                      help='multiplicative factor of learning rate decay')

  parser.add_argument('--train_steps', type=int, default=-
                      1, help='number of iterations on training data for one epoch')
  parser.add_argument('--train_epochs', type=int, default=1,
                      help='number of epochs on training data')
  parser.add_argument(
      '--train_layers_epochs',
      type=int,
      default=-1,
      help='number of epochs to train selected layers before switching to training all layers')
  parser.add_argument(
      '--sampler_type',
      type=str,
      default='rnd',
      choices=[
          'rnd',
          'wc_rnd',
          'seq'],
      help='type of the training data sampler')
  parser.add_argument('--sampler_persistent', action='store_true', help='use a persistent sampler')
  parser.add_argument('--start_epoch', type=int, default=1, help='index of the start epoch')
  parser.add_argument('--input_aug', action='store_true', help='enable input augmentation')
  parser.add_argument('--color_aug', action='store_true', help='enable color augmentation')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
  parser.add_argument(
      '--weighted_pos',
      action='store_true',
      help='weight positive samples for each class in balance with negatives')
  parser.add_argument('--weighted_pos_max', type=float, default=None,
                      help='maximum weight of positive samples for all class')
  parser.add_argument('--eval_steps', type=int, default=-
                      1, help='number of iterations on evaluation data for one epoch')
  parser.add_argument('--eval_batch_size', type=int, default=128, help='batch size for evaluation')
  parser.add_argument('--best_metric', type=str, default='AUC',
                      help='the evaluation metric used to select best model')
  parser.add_argument('--num_workers', type=int, default=0,
                      help='number of workers for data loader')
  parser.add_argument('--seed', type=int, default=-1, help='set random seed')
  parser.add_argument('--no_gpu', action='store_true', help='do not use GPUs')
  # logging
  parser.add_argument(
      '--log_level',
      type=str,
      default=logging.INFO)
  parser.add_argument('--log_interval', type=int, default=100,
                      help='logging interval in terms of iterations')

  tm_start = datetime.now()

  opt = parser.parse_args()

  log_util.config_basic(level=opt.log_level)
  log_file = None

  # Check if save dir exists
  if opt.save_dir is not None:
    if not os.path.exists(opt.save_dir):
      logger.info('creating the save directory at {}'.format(opt.save_dir))
      os.makedirs(opt.save_dir)

    log_file = os.path.join('.', 'log_local.txt')
    log_util.add_file_handler(logger, log_file=log_file, add_to_root=True)

  logger.info('{}'.format(opt))
  logger.info('Torch version: {}'.format(torch.__version__))
  # set random seed
  if opt.seed > 0:
    logger.info('set random seed to <{}>'.format(opt.seed))
    random.seed(opt.seed)
  torch.manual_seed(opt.seed)

  # data
  logger.info('=' * 25)
  # get categories
  logger.info('read categories from csv file {}'.format(opt.category_csv))
  categories_list = datasets.read_category_list(opt.category_csv)
  n_categories = len(categories_list)
  logger.info('number of categories: {}'.format(n_categories))

  # set sample transformers
  image_transform_train = nf.get_default_image_transform(
      augmentation=opt.input_aug, color_augmentation=opt.color_aug)
  image_transform_eval = nf.get_default_image_transform(
    augmentation=False, color_augmentation=False)
  label_transform = transforms.Compose([
      MultiLabelBinarizer(num_classes=n_categories, dtype=torch.float)
  ])
  logger.info('training image transforms: {}'.format(image_transform_train))
  logger.info('evaluation image transforms: {}'.format(image_transform_eval))

  # create a dataset
  logger.info('Create a training EmojiDataset')
  train_ds = datasets.EmojiDataset(categories_list=categories_list, samples_csv_file=opt.train_csv,
                                   input_transform=image_transform_train, target_transform=label_transform, suppress_exceptions=True)
  logger.info('Number of samples in training file: {}'.format(train_ds.n_samples))
  logger.info('Create a validation EmojiDataset')
  valid_ds = datasets.EmojiDataset(categories_list=categories_list, samples_csv_file=opt.val_csv,
                                   input_transform=image_transform_eval, target_transform=label_transform, suppress_exceptions=True)
  logger.info('Number of samples in validation file: {}'.format(valid_ds.n_samples))

  # create data samplers
  logger.info('Create data samplers')
  if opt.sampler_type == 'rnd':
    train_sampler = RandomSampler(train_ds)
  elif opt.sampler_type == 'wc_rnd':
    train_sampler = WeightedClassRandomSampler(train_ds, num_classes=len(categories_list))
  elif opt.sampler_type == 'seq':
    train_sampler = SequentialSampler(train_ds)
  else:
    raise ValueError('unknown training data sampler type <{}>'.format(opt.train_sampler))

  if opt.sampler_persistent:
    logger.info('using a persistent sampler of {}'.format(train_sampler))
    train_sampler = PersistentSampler(train_sampler, train_ds)

  logger.info('Training sampler: {}'.format(type(train_sampler)))
  valid_sampler = SequentialSampler(valid_ds)
  # set batch collate
  collate_fn = partial(
      collate_ignore_nones,
      def_batch=[
          (torch.zeros(
              3,
              nf.INET_IMG_SIZE,
              nf.INET_IMG_SIZE),
              torch.zeros(n_categories))])
  # create loaders
  logger.info('Create data loaders')
  train_dataloader = torch.utils.data.DataLoader(train_ds, sampler=train_sampler,
                                                 batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn)
  eval_dataloader = torch.utils.data.DataLoader(valid_ds, sampler=valid_sampler,
                                                batch_size=opt.eval_batch_size, shuffle=False, num_workers=opt.num_workers, collate_fn=collate_fn, drop_last=False)

  # model
  logger.info('=' * 25)
  # check if there is a gpu
  device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_gpu else 'cpu')
  logger.info('using device: {}'.format(device))
  logger.info('setup model')
  model = nf.setup_model(output_size=n_categories, device=device, opt=opt, load_state=True)
  logger.info('setup optimizer')
  if opt.resume is not None:
    logger.warning(
      'currently loading optimizer state from checkpoint is not supported, creating an optimizer without state intialization')

  optimizer = nf.setup_optimizer(model, opt, load_state=False)
  logger.info('Optimizer: {}'.format(str(optimizer).replace('\n', ' ')))

  lr_scheduler = None
  if opt.scheduler_step_size > 0:
    logger.info('setup learning rate scheduler')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma, last_epoch=-1)

  # Loss
  logger.info('setup loss')
  pos_weight = None
  if opt.weighted_pos:
    logger.info('get positive samples weights for each class from training set')
    pos_weight = train_ds.pos_weights()
    if opt.weighted_pos_max is not None:
      logger.info('set maximum positive weight to {}'.format(opt.weighted_pos_max))
      pos_weight.clamp_(max=opt.weighted_pos_max)

    logger.info('positive weights: \n{}\n'.format(pos_weight))

  loss = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)
  loss = loss.to(device)
  logger.info('Loss: {}'.format(loss))
  # Trainer
  logger.info('Setup Trainer')
  trainer = Trainer(
      model,
      loss,
      optimizer,
      train_dataloader,
      eval_dataloader,
      lr_scheduler=lr_scheduler
  )
  metrics.add_default_train_metrics(trainer, max_k=n_categories - 1)
  metrics.add_default_eval_metrics(trainer, max_k=n_categories - 1)
  if opt.best_metric not in trainer.metric_keys(eval=True):
    raise ValueError(
        'best_metric <{}> is not in the evaluation metrics. Possible values are: {}'.format(
            opt.best_metric,
            trainer.metric_keys(
                eval=True)))

  best_val = None

  # training
  logger.info('=' * 25)
  for epoch in range(opt.start_epoch, opt.train_epochs + opt.start_epoch):
    logger.info('\n')
    # train
    logger.info('Epoch {}: start training...'.format(epoch))
    train_loss, train_metrics = trainer.train(
        max_iter=opt.train_steps, epoch=epoch, device=device, log_interval=opt.log_interval)
    # eval
    logger.info('Epoch {}: start evaluating...'.format(epoch))
    eval_loss, eval_metrics, _ = trainer.eval(
        max_iter=opt.eval_steps, epoch=epoch, device=device, log_interval=5 * opt.log_interval)

    # checkpoint
    is_better, best_val = metrics.is_better(eval_metrics[opt.best_metric], best_val)
    logger.info('save checkpoint')
    nf.save_checkpoint({
        'epoch': epoch,
        'opt': opt,
        'model_state': trainer.model.state_dict(),
        'optimizer_state': trainer.optimizer.state_dict(),
        'train_loss': train_loss,
        'train_metrics': train_metrics,
        'eval_loss': eval_loss,
        'eval_metrics': eval_metrics
    },
        filename=os.path.join(opt.save_dir, 'checkpoint_{:03d}.pth.tar'.format(epoch)),
        is_best=is_better)

    # reset trainable params if needed
    if epoch == opt.train_layers_epochs and len(opt.train_layers) > 0:
      logger.info('\n')
      logger.info('unfreezing all layers in the model to train')
      nf.set_all_layers(trainer.model, freeze=False)
      logger.info(nf.model_info(trainer.model))
      logger.info('reset optimzer')
      trainer.optimizer = nf.get_optimizer(
          opt.optimizer, nf.get_model_parameters(
              trainer.model, only_trainable=True), opt)
      logger.info('Optimizer: {}\n'.format(str(trainer.optimizer).replace('\n', ' ')))
      if trainer.lr_scheduler is not None:
        logger.info('reset scheduler')
        trainer.lr_scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer,
                                                               step_size=opt.scheduler_step_size, gamma=opt.scheduler_gamma, last_epoch=-1)

  logger.info('Run time: {}'.format(datetime.now() - tm_start))
  if log_file is not None and os.path.exists(log_file):
    tms = '_' + datetime.now().strftime('%Y%m%d%H%M%S')
    shutil.copyfile(log_file, os.path.join(opt.save_dir, 'log{}.txt'.format(tms)))

  logger.info('END')
