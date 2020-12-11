from datetime import datetime, timedelta
from itertools import cycle
import logging
import random

from data.datasets import PeekDataset
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler


logger = logging.getLogger(__name__)


class PersistentSampler(Sampler):
  """
  A wrapper of normal sampler where it check if the dataset would return a sample for the selected index.
  If not, the sampler will try to get a different sample for num_trials.
  If all trials fail then the sampler will return None
  """

  def __init__(self, base_sampler, dataset, num_trials=10):
    """
    Parameters
    ----------
    base_sampler: Sampler
            The core sampler wrapped by this class
    dataset: PeekDataset
            The dataset is used to check whether a sample is safe to retrieve using the peek() method
    num_trials: int
            Number of times the sampler will try to retrieve a valid sample before giving up
    """
    if not isinstance(dataset, PeekDataset):
      raise ValueError(
          'dataset should be of type PeekDataset but type {} is found'.format(
              type(dataset)))

    self.sampler = base_sampler
    self.sampler_iter = None
    self.dataset = dataset
    self.num_trials = num_trials
    self._n_failed_samples = 0
    self._t_peeks = timedelta()
    self._n_peeks = 0

  def __iter__(self):
    self.sampler_iter = iter(self.sampler)
    return self

  def __len__(self):
    return len(self.sampler)

  def next(self):
    if self._n_peeks > 0 and self._n_peeks % 10 == 0:
      logging.debug('n_peeks: {}, time_peeks: total= {} per_item= {}, n_failed_samples: {}'.format(
          self._n_peeks, self._t_peeks, self._t_peeks / self._n_peeks, self._n_failed_samples))
    trials = 0
    while trials <= self.num_trials:
      trials += 1
      # sample an index of a data point
      idx = next(self.sampler_iter)
      tm_p = datetime.now()
      self._n_peeks += 1
      # check if it is OK
      is_ok = self.dataset.peek(idx)
      self._t_peeks += (datetime.now() - tm_p)
      if is_ok:
        return idx
      else:
        logger.debug('sample {} not safe to retrieve'.format(idx))
      self._n_failed_samples += 1

    logger.warning('Maximum number of trials reached, failed to retrieve a sample from the dataset.')
    # raise Exception('ERROR: failed to retrieve a sample from the dataset. Maximum number of trials reached')
    return -1


class WeightedClassRandomSampler(Sampler):
  """
  A random sampler that randomly select samples with given class probability / weights
  """

  def __init__(self, dataset, num_classes, weights=None, replacement=True):
    """
    Parameters
    ----------
    dataset: PeekDataset
            The dataset of samples
    num_classes: int
            Number of classes
    weights: float list of length num_classes
            The sampling weight assigned to each class, if None then a uniform distribution is used
    replacement: bool
            Whether to sample with replacement

    """
    if not isinstance(dataset, PeekDataset):
      raise ValueError(
          'dataset should be of type PeekDataset but type {} is found'.format(
              type(dataset)))

    self.dataset = dataset
    self.num_classes = num_classes
    if weights is None:
      logger.info('weights are undefined, sampling using uniform weights for classes')
      self.weights = torch.DoubleTensor(num_classes).fill_(1. / num_classes)
    else:
      self.weights = weights
    logger.debug('class weights: {}'.format(self.weights))
    self.replacement = replacement
    self.class_iter = None
    self.class_samples_iter = []

  def __iter__(self):
    self.class_iter = iter(torch.multinomial(self.weights, len(self.dataset), replacement=True))
    if not self.replacement:
      self.class_samples_iter = []
      for i in range(self.num_classes):
        # self.class_samples_iter.append(cycle(torch.randperm(len(self.dataset.class_sample_ids(i))).tolist()))
        self.class_samples_iter.append(cycle(range(len(self.dataset.class_sample_ids(i)))))

    return self

  def __len__(self):
    return len(self.dataset)

  def next(self):
    # sample a class
    class_id = next(self.class_iter)
    # sample a data point index
    if not self.replacement:
      x_id = next(self.class_samples_iter[class_id])
    else:
      x_id = random.randint(0, len(self.dataset.class_sample_ids(class_id)) - 1)

    logger.debug('sample from class {} data point {}'.format(class_id, x_id))
    return self.dataset.class_sample_ids(class_id)[x_id]


def collate_ignore_nones(batch, def_batch=None):
  """
  Given a batch of samples, removes any sample with None values
  """
  # remove any sample that has None in it
  batch_without_nones = list(filter(lambda x: not any(y is None for y in x), batch))
  logger.debug('input batch: {}, output batch: {}'.format(len(batch), len(batch_without_nones)))
  if len(batch_without_nones) == 0:
    logger.warning('input batch is all Nones, return default batch')
    batch_without_nones = def_batch

  # apply default_collate on the filtered batch
  batch_out = default_collate(batch_without_nones)
  return batch_out
