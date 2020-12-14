"""
Copyright 2020 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""
import ast
import io
import logging
import requests

from PIL import Image, ImageFile
import pandas as pd
import torch
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


COLUMN_NAME_LABELS = 'labels'
COLUMN_NAME_URL = 'url'
COLUMN_NAME_CATEGORY = 'category'


logger = logging.getLogger(__name__)


def read_category_list(csv_file, column_name=COLUMN_NAME_CATEGORY):
  with open(csv_file) as f:
    categories = pd.read_csv(f, encoding='utf-8', engine='python')

  categories_list = categories[column_name].values.tolist()
  return categories_list


class PeekDataset(Dataset):
  """
  A wrapper for base pytorch Dataset class that has a peek method to check if a sample is safe to fetch
  """

  def __init__(self):
    super(PeekDataset, self).__init__()

  def class_sample_ids(self, class_id):
    """
    Returns a list of sample ids that are labeled with class_id
    """
    return []

  def peek(self, index):
    """
    Checks if the sample at position <index> is safe to retrieve
    """
    return True


class EmojiDataset(PeekDataset):
  """
  A dataset class for emoji prediction / multi-label classification
  """

  def __init__(self,
               samples_csv_file,
               categories_list,
               input_transform=None,
               target_transform=None,
               suppress_exceptions=False,
               cl_labels=COLUMN_NAME_LABELS,
               cl_url=COLUMN_NAME_URL):
    """
    Parameters
    ----------
    samples_csv_file: str
            csv file name with data samples, has columns (cl_labels, cl_url)
    categories_list: list
            list of category names where index matches category label in csv
    input_transform: pytorch transformer
            a transformer for input data (images)
    target_transform: pytorch transformer
            a transformer for target data (labels)
    supress_exception: bool
            if True then exceptions while reading images are caught and return None otherwise an exception is raised
    """

    self.suppress_exceptions = suppress_exceptions
    # set categories information
    self.num_categories = len(categories_list)
    self.category_name = categories_list
    self.category_id = {}
    for i in range(len(self.category_name)):
      self.category_id[self.category_name[i]] = i
    self.cl_labels = cl_labels
    self.cl_url = cl_url

    # read samples from csv
    sp_column_names = [cl_labels, cl_url]
    logger.info('read samples from csv file {}'.format(samples_csv_file))
    with open(samples_csv_file) as f:
      self.samples = pd.read_csv(f, usecols=sp_column_names, encoding='utf-8', engine='python')

    logger.info('number of samples loaded: {}'.format(len(self.samples)))
    self.samples[cl_labels] = self.samples[cl_labels].apply(ast.literal_eval)

    # get a list of sample ids for each class
    logger.info('compile a list of samples ids for each class in the dataset')
    self.class_samples = []
    for i in range(self.num_categories):
      self.class_samples.append(
        self.samples.index[self.samples[cl_labels].map(lambda lbls: i in lbls)].values)

    self.input_transform = input_transform
    self.target_transform = target_transform

  @property
  def n_categories(self):
    return self.num_categories

  @property
  def n_samples(self):
    return len(self.samples)

  def __len__(self):
    return len(self.samples)

  def get_url(self, index):
    url = self.samples[self.cl_url][index]
    return url

  def _get_image_from_url(self, url):
    """ Fetch image using input url """
    try:
      return requests.get(url).content
    except Exception as exception:
      return exception

  def peek(self, index):
    """ Checks if the sample is good, i.e. image exist """
    logger.debug('peek on sample {}'.format(index))
    try:
      url = self.samples[self.cl_url][index]
      img_bytes = self._get_image_from_url(url)
      if not isinstance(img_bytes, Exception):
        return True
      else:
        return False
    except Exception:
      return False

  def class_sample_ids(self, class_id):
    """ Returns a list of sample ids for the given class """
    return self.class_samples[class_id]

  def __getitem__(self, index):
    if index is None:
      return None, None
    # get image from url
    url = self.samples[self.cl_url][index]
    logger.debug('getting sample {} from url {}'.format(index, url))
    img = None
    try:
      img_bytes = self._get_image_from_url(url)
      if not isinstance(img_bytes, Exception):
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')  # make sure it's a 3 channel image
      else:
        if self.suppress_exceptions:
          return None, None
        else:
          raise Exception('failed to fetch image {} from url {}'.format(index, img_bytes))

    except Exception as exc:
      logger.debug('failed to get {}: {}'.format(index, exc))

    # get labels
    lbls = self.samples[self.cl_labels][index]

    # transform sample
    if self.input_transform and img is not None:
      img = self.input_transform(img)
    if self.target_transform and lbls is not None:
      lbls = self.target_transform(lbls)

    return img, lbls, index

  def pos_weights(self):
    n_samples = len(self.samples)
    bin_lbls = []
    # iterate over samples and collect binary labels
    for i in range(n_samples):
      lbli = self.samples[self.cl_labels][i]
      one_hot = torch.FloatTensor(self.num_categories).zero_()
      if len(lbli) > 0:
        one_hot.scatter_(0, torch.LongTensor(lbli), 1)

      bin_lbls.append(one_hot.view(1, -1))

    lbls_all = torch.cat(bin_lbls, dim=0)
    lbls_pos = lbls_all.sum(dim=0)
    lbls_neg = n_samples - lbls_pos
    pos_wg = lbls_neg / lbls_pos
    return pos_wg
