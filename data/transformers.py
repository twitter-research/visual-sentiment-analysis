"""
Copyright 2020 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""
from __future__ import division

import logging
import math

import torch
import torchvision.transforms.functional as F


logger = logging.getLogger(__name__)


class MultiLabelBinarizer(object):
  """
  A transformer that take in a list of integer labels and returns a one-hot tensor
  """

  def __init__(self, num_classes, dtype=torch.uint8):
    """
    Parameter
    ---------
    num_classes: int
            Number of classes
    dtype: torch.dtype
            Type of returned one-hot tensor
    """
    self.num_classes = num_classes
    self.dtype = dtype

  def __call__(self, labels):
    if torch.is_tensor(labels):
      labels = labels.tolist()

    if not isinstance(labels, list):
      labels = [labels]

    labels = torch.LongTensor(labels)
    one_hot = torch.FloatTensor(self.num_classes).zero_()
    if len(labels) > 0:
      try:
        one_hot.scatter_(0, labels, 1)
      except Exception as exc:
        logger.error('Error in scatter! labels {}, ExcMsg: {}'.format(labels, exc))

    one_hot = one_hot.to(self.dtype)
    return one_hot

  def __repr__(self):
    return self.__class__.__name__ + '(num_classes={}, dtype={})'.format(self.num_classes, self.dtype)


class SquareImage(object):
  """
  A transformer that takes in a PIL image and returns a PIL image with equal width and height by padding the smaller dimension
  """

  def __init__(self, fill=0, padding_mode='constant'):
    super(SquareImage, self).__init__()
    self.color = fill
    self.mode = padding_mode

  def __call__(self, image):
    width, height = image.size
    m_dim = max(width, height)
    padding = (
        int(math.ceil((m_dim - width) / 2)),
        int(math.ceil((m_dim - height) / 2)),
        int(math.floor((m_dim - width) / 2)),
        int(math.floor((m_dim - height) / 2))
    )
    im = F.pad(image, padding=padding, fill=self.color, padding_mode=self.mode)
    return im


class ResizeLargest(object):
  """
  A transformer that takes in a PIL image and returns a resized PIL image with the larger dimension equal to <size>\
  """

  def __init__(self, size, interpolation=2):
    super(ResizeLargest, self).__init__()
    self.size = int(size)
    self.interpolation = interpolation

  def __call__(self, image):
    width, height = image.size
    if height > width:
      new_size = (self.size, int(self.size * (width / height)))
    else:
      new_size = (int(self.size * (height / width)), self.size)

    im = F.resize(image, size=new_size, interpolation=self.interpolation)
    return im
