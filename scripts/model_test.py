"""
Copyright 2020 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""
from __future__ import division

import argparse
from datetime import datetime
from functools import partial
import shutil
import os
import random
import logging

import data.datasets as datasets
from data.samplers import collate_ignore_nones
from data.transformers import MultiLabelBinarizer
from model import nn_factory as nf
import model.metrics as metrics
from model.train import Trainer
from utils import log as log_util

import torch
import torchvision.transforms as transforms


logger = logging.getLogger(__name__)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # data
  parser.add_argument('--test_csv', type=str, default=None, help='csv file with testing samples')
  parser.add_argument('--category_csv', type=str, default=None, help='csv file with category names')
  parser.add_argument('--save_dir', type=str, default=None, help='save directory')
  # model
  parser.add_argument('--model_file', type=str, default=None, help='model file path')
  parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
  parser.add_argument('--test_steps', type=int, default=-
                      1, help='number of iterations on evaluation data for one epoch')
  parser.add_argument('--num_workers', type=int, default=0,
                      help='number of workers for data loader')
  parser.add_argument('--seed', type=int, default=-1, help='set random seed')
  parser.add_argument('--no_gpu', action='store_true', help='do not use GPUs')
  # qualitative results
  parser.add_argument(
      '--no_qualitative',
      action='store_true',
      help='disable qualitative evaluation')
  parser.add_argument('--per_class_samples', type=int, default=20,
                      help='the number of most confident samples per class to visualize')
  parser.add_argument('--multilabel_samples', type=int, default=30,
                      help='the number of most confident samples across all labels to visualize')
  parser.add_argument('--multilabel_k', type=int, default=5,
                      help='top k criteria used to rank multilabel samples')
  parser.add_argument('--image_size', type=int, default=256,
                      help='image size for qualitative results')
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

  # check if model file exists
  if not os.path.exists(opt.model_file):
    raise ValueError('model file not found! {}'.format(opt.model_file))

  log_file = None
  if opt.save_dir is not None:
    if not os.path.exists(opt.save_dir):
      logger.info('creating the save directory at {}'.format(opt.save_dir))
      os.makedirs(opt.save_dir)
      if not os.path.exists(opt.save_dir):
        raise IOError('can not create the save directory!!')
      else:
        logger.info('directory successfully created')

    log_file = os.path.join('.', 'log_model_test.txt')
    log_util.add_file_handler(logger, log_file=log_file, add_to_root=True)

  logger.info('{}'.format(opt))
  logger.info('Torch version: {}'.format(torch.__version__))
  # set random seed
  if opt.seed > 0:
    logger.info('set random seed to <{}>'.format(opt.seed))
    random.seed(opt.seed)
  torch.manual_seed(opt.seed)

  # set tensorboardX
  log_dir = './plogs_test'
  if opt.save_dir is not None:
    log_dir = opt.save_dir

  # data
  logger.info('=' * 25)
  # get categories
  logger.info('read categories from csv file {}'.format(opt.category_csv))
  categories_list = datasets.read_category_list(opt.category_csv)
  n_categories = len(categories_list)
  logger.info('number of categories: {}'.format(n_categories))

  # set sample transformers
  image_transform = nf.get_default_image_transform(augmentation=False)
  label_transform = transforms.Compose([
      MultiLabelBinarizer(num_classes=n_categories, dtype=torch.float)
  ])
  logger.info('image transforms: {}'.format(image_transform))

  # create a dataset
  logger.info('Create a testing EmojiDataset')
  test_ds = datasets.EmojiDataset(categories_list=categories_list, samples_csv_file=opt.test_csv,
                                  input_transform=image_transform, target_transform=label_transform, suppress_exceptions=True)
  logger.info('Number of samples in testing file: {}'.format(test_ds.n_samples))

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
  test_dataloader = torch.utils.data.DataLoader(
      test_ds,
      batch_size=opt.batch_size,
      shuffle=False,
      num_workers=opt.num_workers,
      collate_fn=collate_fn)

  # model
  logger.info('=' * 25)
  logger.info('loading model from file')
  checkpoint = nf.load_checkpoint(opt.model_file)
  logger.info('in checkpoint: {}'.format(checkpoint.keys()))
  model_name = checkpoint['opt'].net_name
  logger.info('model type: {}'.format(model_name))
  model = nf.create_and_init_model(model_name, checkpoint['model_state'], output_size=n_categories)
  # check if there is a gpu
  device = torch.device('cuda' if torch.cuda.is_available() and not opt.no_gpu else 'cpu')
  logger.info('using device: {}'.format(device))
  model = model.to(device)
  logger.info('model evaluation loss: {}'.format(checkpoint['eval_loss']))
  logger.info('model evaluation metrics: ')
  for k in checkpoint['eval_metrics'].keys():
    logger.info('{}: {}'.format(k, checkpoint['eval_metrics'][k]))
    if torch.is_tensor(checkpoint['eval_metrics'][k]) and checkpoint['eval_metrics'][k].numel() > 1:
      logger.info('{} mean: {:.4f}'.format(k, checkpoint['eval_metrics'][k].mean()))

  # Loss
  loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
  loss = loss.to(device)

  # tester
  logger.info('Setup Tester')
  tester = Trainer(model, loss, optimizer=None, train_data_loader=None,
                   eval_data_loader=test_dataloader)
  metrics.add_default_eval_metrics(tester, max_k=n_categories - 1)
  if not opt.no_qualitative:
    logger.info('add qualitative metrics')
    tester.add_metric(
        'TopBPred',
        metrics.TopBinaryPredictions(
            n_samples=opt.per_class_samples),
        eval=True)
    tester.add_metric(
        'TopMPred',
        metrics.TopMultiLabelPredictions(
            n_samples=opt.multilabel_samples,
            k=opt.multilabel_k),
        eval=True)

  # testing
  epoch = checkpoint['epoch']
  logger.info('start testing for model at epoch {}'.format(epoch))
  test_loss, test_metrics, index_map = tester.eval(
      max_iter=opt.test_steps, epoch=epoch, device=device, log_interval=opt.log_interval)

  logger.info('test loss: {}'.format(test_loss))
  logger.info('test metrics: \n')
  for k in test_metrics.keys():
    if k not in ['TopBPred', 'TopMPred']:
      logger.info('{}: {}'.format(k, test_metrics[k]))
      if torch.is_tensor(test_metrics[k]) and test_metrics[k].numel() > 1:
        logger.info('{} mean: {:.4f}'.format(k, test_metrics[k].mean()))

  # save to file
  model_fname = os.path.basename(opt.model_file).split('.')[0]
  outfile = os.path.join(opt.save_dir, '{}.test.pth'.format(model_fname))
  logger.info('save test results to {}'.format(outfile))
  with open(outfile, mode='wb') as f:
    torch.save({
        'opt': opt,
        'test_loss': test_loss,
        'test_metrics': test_metrics,
        'index_map': index_map
    }, f)

  logger.info('Run time: {}'.format(datetime.now() - tm_start))
  if log_file is not None and os.path.exists(log_file):
    tms = '_' + datetime.now().strftime('%Y%m%d%H%M%S')
    shutil.copy(log_file, os.path.join(opt.save_dir, 'log_test{}.txt'.format(tms)))

  logger.info('END')
