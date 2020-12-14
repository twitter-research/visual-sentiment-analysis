"""
Copyright 2020 Twitter, Inc.
Licensed under the Apache License, Version 2.0
http://www.apache.org/licenses/LICENSE-2.0
"""
from __future__ import division

import logging
import math
import numbers

import numpy as np
import torch


logger = logging.getLogger(__name__)


def is_better(metric_val, metric_best=None):
  if metric_best is None:
    metric_best = metric_val.clone()
    is_better = True
    return is_better, metric_best

  if torch.is_tensor(metric_val) and len(metric_val) > 1:
    # mulit-value metric
    n_better = (metric_val >= metric_best).sum().item()
    if n_better > (len(metric_best) * 0.5):
      metric_best = metric_val.clone()
      is_better = True
      logger.info('This model is better, it improves {}/{} of the classes'.format(
          n_better, len(metric_best)))
      return is_better, metric_best
  else:
    # single-value metric
    if metric_val > metric_best:
      metric_best = metric_val
      is_better = True
      return is_better, metric_best

  return False, metric_best


def add_default_eval_metrics(trainer, max_k=10):
  trainer.add_metric('mAP', MeanAP(), eval=True)
  trainer.add_metric('AUC', Auc(), eval=True)
  if max_k >= 1:
    trainer.add_metric('Top1', TOPkMultiLabel(k=1), eval=True)
  if max_k >= 3:
    trainer.add_metric('Top3', TOPkMultiLabel(k=3), eval=True)
  if max_k >= 5:
    trainer.add_metric('Top5', TOPkMultiLabel(k=5), eval=True)
  if max_k >= 10:
    trainer.add_metric('Top10', TOPkMultiLabel(k=10), eval=True)


def add_default_train_metrics(trainer, max_k=5):
  trainer.add_metric('label_dist', LabelDistribution(), eval=False)
  if max_k >= 1:
    trainer.add_metric('Top1', TOPkMultiLabel(k=1), eval=False)
  if max_k >= 3:
    trainer.add_metric('Top3', TOPkMultiLabel(k=3), eval=False)
  if max_k >= 5:
    trainer.add_metric('Top5', TOPkMultiLabel(k=5), eval=False)


class Metric(object):
  def __init__(self):
    super(Metric, self).__init__()

  def reset(self):
    pass

  def update(self, output, target):
    pass

  def compute(self):
    pass


class MeanAP(Metric):
  """
  Calculates the mean average precision.
  from torchnet APMeter: https://github.com/pytorch/tnt/blob/master/torchnet/meter/apmeter.py
  """

  def __init__(self):
    super(MeanAP, self).__init__()
    self.reset()

  def reset(self):
    """
    Resets the meter with empty member variables
    """
    self.scores = torch.FloatTensor(torch.FloatStorage())
    self.targets = torch.LongTensor(torch.LongStorage())
    self.weights = torch.FloatTensor(torch.FloatStorage())

  def update(self, output, target, weight=None):
    """
    Add a new observation
    Args:
            output (Tensor): NxK tensor that for each of the N examples
                    indicates the probability of the example belonging to each of
                    the K classes, according to the model. The probabilities should
                    sum to one over all classes
            target (Tensor): binary NxK tensort that encodes which of the K
                    classes are associated with the N-th input
                    (eg: a row [0, 1, 0, 1] indicates that the example is
                    associated with classes 2 and 4)
            weight (optional, Tensor): Nx1 tensor representing the weight for
                    each example (each weight > 0)
    """
    if not torch.is_tensor(output):
      output = torch.from_numpy(output)
    if not torch.is_tensor(target):
      target = torch.from_numpy(target)

    if weight is not None:
      if not torch.is_tensor(weight):
        weight = torch.from_numpy(weight)
      weight = weight.squeeze()

    if output.dim() == 1:
      output = output.view(-1, 1)
    else:
      assert output.dim() == 2, 'wrong output size (should be 1D or 2D with one column per class)'

    if target.dim() == 1:
      target = target.view(-1, 1)
    else:
      assert target.dim() == 2, 'wrong target size (should be 1D or 2D with one column per class)'

    if weight is not None:
      assert weight.dim() == 1, 'Weight dimension should be 1'
      assert weight.numel() == target.size(0), 'Weight dimension 1 should be the same as that of target'
      assert torch.min(weight) >= 0, 'Weight should be non-negative only'

    assert torch.equal(target**2, target), 'targets should be binary (0 or 1)'
    if self.scores.numel() > 0:
      assert target.size(1) == self.targets.size(
        1), 'dimensions for output should match previously added examples.'

    # make sure storage is of sufficient size
    if self.scores.storage().size() < self.scores.numel() + output.numel():
      new_size = math.ceil(self.scores.storage().size() * 1.5)
      new_weight_size = math.ceil(self.weights.storage().size() * 1.5)
      self.scores.storage().resize_(int(new_size + output.numel()))
      self.targets.storage().resize_(int(new_size + output.numel()))
      if weight is not None:
        self.weights.storage().resize_(int(new_weight_size + output.size(0)))

    # store scores and targets
    offset = self.scores.size(0) if self.scores.dim() > 0 else 0
    self.scores.resize_(offset + output.size(0), output.size(1))
    self.targets.resize_(offset + target.size(0), target.size(1))
    self.scores.narrow(0, offset, output.size(0)).copy_(output)
    self.targets.narrow(0, offset, target.size(0)).copy_(target)

    if weight is not None:
      self.weights.resize_(offset + weight.size(0))
      self.weights.narrow(0, offset, weight.size(0)).copy_(weight)

  def compute(self):
    """
    Returns the model's average precision for each class
    Return:
            ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if self.scores.numel() == 0:
      return 0

    ap = torch.zeros(self.scores.size(1))
    if hasattr(torch, "arange"):
      rg = torch.arange(1, self.scores.size(0) + 1).float()
    else:
      rg = torch.range(1, self.scores.size(0)).float()

    if self.weights.numel() > 0:
      weight = self.weights.new(self.weights.size())
      weighted_truth = self.weights.new(self.weights.size())

    # compute average precision for each class
    for k in range(self.scores.size(1)):
      # sort scores
      scores = self.scores[:, k]
      targets = self.targets[:, k]
      _, sortind = torch.sort(scores, 0, True)
      truth = targets[sortind]
      if self.weights.numel() > 0:
        weight = self.weights[sortind]
        weighted_truth = truth.float() * weight
        rg = weight.cumsum(0)

      # compute true positive sums
      if self.weights.numel() > 0:
        tp = weighted_truth.cumsum(0)
      else:
        tp = truth.float().cumsum(0)

      # compute precision curve
      precision = tp.div(rg)

      # compute average precision
      ap[k] = precision[truth.byte()].sum() / max(float(truth.sum()), 1)

    return ap


class LabelDistribution(Metric):
  def __init__(self, ndim=0):
    super(LabelDistribution, self).__init__()
    self.ndim = ndim
    self.reset()

  def reset(self):
    self.targets = torch.LongTensor(self.ndim).fill_(0)
    self.n_samples = 0

  def update(self, output, target):
    if target.dim() == 1:
      target.view(1, -1)

    if len(self.targets) == 0:
      self.targets.resize_(target.size(1)).fill_(0)

    self.n_samples += target.size(0)
    self.targets = self.targets + target.sum(dim=0).type(self.targets.type())

  def compute(self):
    out = self.targets.float() / self.n_samples
    return out


class Roc(Metric):
  """
  Claculates ROC curve
  from torchnet AUCMeter: https://github.com/pytorch/tnt/blob/master/torchnet/meter/aucmeter.py
  """

  def __init__(self):
    super(Roc, self).__init__()
    self.reset()

  def reset(self):
    self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
    self.targets = torch.LongTensor(torch.LongStorage()).numpy()

  def update(self, output, target):
    if torch.is_tensor(output):
      output = output.cpu().squeeze().numpy()

    if torch.is_tensor(target):
      target = target.cpu().squeeze().numpy()
    elif isinstance(target, numbers.Number):
      target = np.asarray([target])
    else:
      logger.error('unkown type of target')

    assert np.ndim(output) == 1, 'wrong output size (1D expected)'
    assert np.ndim(target) == 1, 'wrong target size (1D expected)'
    assert output.shape[0] == target.shape[0], 'number of outputs and targets does not match'
    assert np.all(
        np.add(
            np.equal(
                target, 1), np.equal(
                target, 0))), 'targets should be binary (0, 1)'

    self.scores = np.append(self.scores, output)
    self.targets = np.append(self.targets, target)

  def compute(self):
    # case when number of elements added are 0
    if self.scores.shape[0] == 0:
      return 0.5

    # sorting the arrays
    scores, sortind = torch.sort(torch.from_numpy(self.scores), dim=0, descending=True)
    scores = scores.numpy()
    sortind = sortind.numpy()

    # creating the roc curve
    tpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)
    fpr = np.zeros(shape=(scores.size + 1), dtype=np.float64)

    for i in range(1, scores.size + 1):
      if self.targets[sortind[i - 1]] == 1:
        tpr[i] = tpr[i - 1] + 1
        fpr[i] = fpr[i - 1]
      else:
        tpr[i] = tpr[i - 1]
        fpr[i] = fpr[i - 1] + 1

    tpr /= (self.targets.sum() * 1.0)
    fpr /= ((self.targets - 1.0).sum() * -1.0)

    # calculating area under curve using trapezoidal rule
    n = tpr.shape[0]
    h = fpr[1:n] - fpr[0:n - 1]
    sum_h = np.zeros(fpr.shape)
    sum_h[0:n - 1] = h
    sum_h[1:n] += h
    area = (sum_h * tpr).sum() / 2.0

    return (area, tpr, fpr)


class Auc(Metric):
  def __init__(self):
    super(Auc, self).__init__()
    self.roc = []
    self.reset()

  def reset(self):
    if len(self.roc) > 0:
      for i in range(len(self.roc)):
        self.roc[i].reset()

  def update(self, output, target):
    if len(self.roc) == 0:
      n_classes = output.size(1)
      for i in range(n_classes):
        self.roc.append(Roc())
    for i in range(len(self.roc)):
      self.roc[i].update(output[:, i], target[:, i])

  def compute(self):
    aucs = []
    for i in range(len(self.roc)):
      auci, _, _ = self.roc[i].compute()
      aucs.append(auci)

    out = torch.FloatTensor(aucs)
    return out


class TOPkMultiLabel(Metric):
  def __init__(self, k=1):
    super(TOPkMultiLabel, self).__init__()
    self.k = k
    self.reset()

  def reset(self):
    self.correct = []

  def update(self, output, target):
    if output.dim() == 1:
      output.view_(1, -1)
    if target.dim() == 1:
      target.view_(1, -1)

    assert output.size(0) == target.size(0), 'output and target should have the same size at dim=0'
    outk_score, outk_idx = output.topk(k=self.k, dim=1, largest=True, sorted=True)
    outk_idx = outk_idx.tolist()
    for i in range(target.size(0)):
      tar_idx = target[i, :].nonzero()
      if len(tar_idx) > 0:
        tar_idx = set(tar_idx.squeeze(dim=1).tolist())
        # corr_p = len(set(outk_idx[i]).intersection(tar_idx)) / len(tar_idx)
        corr_p = len(set(outk_idx[i]).intersection(tar_idx)) / min(self.k, len(tar_idx))
        self.correct.append(corr_p)
        # if corr_p > 0.9:
        # 	logger.info('TOPk: {}, Predicted: {}, Label: {}'.format(corr_p, outk_idx[i], tar_idx))

      else:
        self.correct.append(0.0)

  def compute(self):
    out = torch.FloatTensor(self.correct)
    return out.mean()


class TopBinaryPredictions(Metric):
  def __init__(self, n_samples=20):
    super(TopBinaryPredictions, self).__init__()
    self.n_samples = n_samples
    self.reset()

  def reset(self):
    self.outputs = []
    self.targets = []

  def update(self, output, target):
    self.outputs.append(output)
    self.targets.append(target)

  def compute(self):
    outputs = torch.cat(self.outputs, dim=0)
    outputs = outputs.cpu().float()
    targets = torch.cat(self.targets, dim=0)
    targets = targets.cpu().byte()
    top_scores, top_ids = torch.topk(outputs, self.n_samples, dim=0, largest=True, sorted=True)
    top_labels = torch.zeros(self.n_samples, outputs.size(1), dtype=targets.dtype)
    for ci in range(targets.size(1)):
      top_labels[:, ci] = targets[top_ids[:, ci], ci]

    return top_ids, top_scores, top_labels


class TopMultiLabelPredictions(Metric):
  def __init__(self, n_samples=20, k=5):
    super(TopMultiLabelPredictions, self).__init__()
    self.k = k
    self.n_samples = n_samples
    self.topk_mlbl = TOPkMultiLabel(k=k)
    self.reset()

  def reset(self):
    self.outputs = []
    self.targets = []
    self.topk_mlbl.reset()

  def update(self, output, target):
    self.outputs.append(output)
    self.targets.append(target)
    self.topk_mlbl.update(output, target)

  def compute(self):
    correct = torch.FloatTensor(self.topk_mlbl.correct)
    # logger.info('topk correct: {}'.format(correct.tolist()))
    # logger.info('topk: {}'.format(correct.mean()))
    outputs = torch.cat(self.outputs, dim=0)
    outputs = outputs.cpu().float()
    targets = torch.cat(self.targets, dim=0)
    targets = targets.cpu().byte()

    topk_corr, top_ids = torch.topk(correct, k=self.n_samples, dim=0)
    top_scores = outputs[top_ids, :]
    topk_scores, topk_ids = torch.topk(top_scores, k=self.k, dim=1)
    top_labels = targets[top_ids, :]

    return top_ids, topk_scores, topk_ids, topk_corr, top_labels


class ConfusionMatrix(Metric):
  """
  Calculates confusion matrix
  from torchnet ConfusionMeter: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
  Args:
          k (int): number of classes in the classification problem
          normalized (boolean): Determines whether or not the confusion matrix is normalized or not
  """

  def __init__(self, k, normalized=False):
    super(ConfusionMatrix, self).__init__()
    self.conf = np.ndarray((k, k), dtype=np.int32)
    self.normalized = normalized
    self.k = k
    self.reset()

  def reset(self):
    self.conf.fill(0)

  def update(self, predicted, target):
    """Computes the confusion matrix of K x K size where K is no of classes
    Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from the model for N examples and K classes or an N-tensor of integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer values between 0 and K-1 or N x K tensor, where targets are assumed to be provided as one-hot vectors
    """
    predicted = predicted.cpu().numpy()
    target = target.cpu().numpy()

    assert predicted.shape[0] == target.shape[0], 'number of targets and predicted outputs do not match'

    if np.ndim(predicted) != 1:
      assert predicted.shape[1] == self.k, 'number of predictions does not match size of confusion matrix'
      predicted = np.argmax(predicted, 1)
    else:
      assert (predicted.max() < self.k) and (predicted.min() >= 0), 'predicted values are not between 1 and k'

    onehot_target = np.ndim(target) != 1
    if onehot_target:
      assert target.shape[1] == self.k, 'Onehot target does not match size of confusion matrix'
      assert (target >= 0).all() and (target <= 1).all(), 'in one-hot encoding, target values should be 0 or 1'
      assert (target.sum(1) == 1).all(), 'multi-label setting is not supported'
      target = np.argmax(target, 1)
    else:
      assert (predicted.max() < self.k) and (predicted.min() >= 0), 'predicted values are not between 0 and k-1'

    # hack for bincounting 2 arrays together
    x = predicted + self.k * target
    bincount_2d = np.bincount(x.astype(np.int32), minlength=self.k ** 2)
    assert bincount_2d.size == self.k ** 2
    conf = bincount_2d.reshape((self.k, self.k))

    self.conf += conf

  def compute(self):
    """
    Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
    """
    if self.normalized:
      conf = self.conf.astype(np.float32)
      conf = conf / conf.sum(1).clip(min=1e-12)[:, None]
    else:
      conf = self.conf

    return torch.FloatTensor(conf)
