from datetime import datetime
import logging
import time

import torch

logger = logging.getLogger(__name__)


class Trainer(object):
  def __init__(self, model, loss, optimizer, train_data_loader,
               eval_data_loader, plotter=None, lr_scheduler=None):
    self.model = model
    self.optimizer = optimizer
    self.lr_scheduler = lr_scheduler
    self.train_data_loader = train_data_loader
    self.eval_data_loader = eval_data_loader
    self.loss = loss
    self.plotter = plotter
    self.metrics_eval = {}
    self.metrics_train = {}
    self.metrics_output_transform = None
    self.metrics_target_transform = None

  def train(self, max_iter=-1, device='cpu', epoch=0, log_interval=10):
    n_batches = len(self.train_data_loader)
    if max_iter > 0:
      n_batches = min(max_iter, n_batches)

    if self.lr_scheduler is not None:
      self.lr_scheduler.step(epoch=epoch)
      logger.info('step in lr_scheduler, current lr: {}'.format(self.lr_scheduler.get_lr()))

    logger.debug(
        'loader, in_trans: {}, tar_trans: {}'.format(
            self.train_data_loader.dataset.input_transform,
            self.train_data_loader.dataset.target_transform))
    # set model for training
    self.model.train()
    # iterate over training data
    tm_epoch = datetime.now()
    tm_data = time.time()
    self.reset_metrics(eval=False)
    train_loss = 0
    n_samples = 0
    for batch_idx, batch_data in enumerate(self.train_data_loader):
      tm_data_el = time.time() - tm_data
      data, target = batch_data[0].to(device), batch_data[1].to(device)
      logger.debug('input size: {}, target size: {}'.format(data.size(), target.size()))
      tm_batch = time.time()
      self.optimizer.zero_grad()
      # forward pass
      output = self.model(data)
      logger.debug('output size: {}'.format(output.size()))
      loss = self.loss(output, target)
      train_loss += loss.item()
      # backward pass
      loss.backward()
      # optimize
      self.optimizer.step()
      # logging
      n_samples += len(data)
      self.update_metrics(output, target, eval=False)
      if batch_idx % log_interval == 0:
        logger.info('E_{:02d} ({:5.2f}%) [{:5d}/{}]\tLoss: {:.6f}\tTime: D[{:7.3f}] B[{:7.3f}]\tBSz: {:4.1f}%'.format(
            epoch,
            100. * batch_idx / n_batches,
            n_samples, n_batches * self.train_data_loader.batch_size,
            loss.item(),
            tm_data_el, time.time() - tm_batch,
            100. * data.size(0) / self.train_data_loader.batch_size))

        if self.plotter is not None:
          global_idx = (batch_idx) + ((epoch - 1) * n_batches)
          self.plotter.add_batch_loss(loss.item(), global_idx)

      tm_data = time.time()
      if batch_idx >= n_batches:
        break

    if self.loss.reduction == 'elementwise_mean':
      train_loss /= n_batches
    else:
      train_loss /= n_samples

    n_all_samples = n_batches * self.train_data_loader.batch_size
    logger.info('E_{:02d} Training Average loss: {:.6f}, missed samples: {:.2f}%, finished in {}\n'.format(
        epoch,
        train_loss,
        100.0 * (n_all_samples - n_samples) / n_all_samples,
        datetime.now() - tm_epoch))

    metric_train = self.compute_metrics(eval=False)
    return train_loss, metric_train

  def eval(self, max_iter=-1, device='cpu', epoch=0, log_interval=10):
    n_batches = len(self.eval_data_loader)
    if max_iter > 0:
      n_batches = min(max_iter, n_batches)

    # set model for evaluation
    self.model.eval()
    self.reset_metrics(eval=True)
    eval_loss = 0
    n_samples = 0
    tm_epoch = datetime.now()
    index_map = []
    with torch.no_grad():
      for batch_idx, batch_data in enumerate(self.eval_data_loader):
        data, target = batch_data[0].to(device), batch_data[1].to(device)
        index_map.append(batch_data[2])
        output = self.model(data)
        eval_loss += self.loss(output, target).item()
        n_samples += len(data)
        self.update_metrics(output, target, eval=True)
        if batch_idx % log_interval == 0:
          logger.info('E_{:02d} ({:5.2f}%) [{:5d}/{}] remaining time {}'.format(
              epoch,
              100. * batch_idx / n_batches,
              n_samples, n_batches * self.eval_data_loader.batch_size,
              ((datetime.now() - tm_epoch) / (batch_idx + 1)) * (n_batches - batch_idx)))
        if batch_idx >= n_batches:
          break

    if self.loss.reduction == 'elementwise_mean':
      eval_loss /= n_batches
    else:
      eval_loss /= n_samples

    n_all_samples = n_batches * self.eval_data_loader.batch_size
    logger.info('E_{:02d} Evaluation Average loss: {:.6f}, missed samples: {:.2f}%, finished in {}\n'.format(
        epoch,
        eval_loss,
        100.0 * (n_all_samples - n_samples) / n_all_samples,
        datetime.now() - tm_epoch))
    # get performance metrics
    metric_val = self.compute_metrics(eval=True)
    return eval_loss, metric_val, torch.cat(index_map, dim=0)

  def add_metric(self, label, metric, eval=True):
    if eval:
      self.metrics_eval[label] = metric
    else:
      self.metrics_train[label] = metric

  def reset_metrics(self, eval=True):
    if eval:
      for k in self.metrics_eval.keys():
        self.metrics_eval[k].reset()
    else:
      for k in self.metrics_train.keys():
        self.metrics_train[k].reset()

  def update_metrics(self, output, target, eval=True):
    if self.metrics_output_transform is not None:
      outputs = []
      for i in range(target.size(0)):
        outputs.append(self.metrics_output_transform(output[i]).view(1, -1))

      output = torch.cat(outputs, dim=0)

    if self.metrics_target_transform is not None:
      targets = []
      for i in range(target.size(0)):
        targets.append(self.metrics_target_transform(target[i]).view(1, -1))

      target = torch.cat(targets, dim=0)

    if output.dim() == 1:
      output = output.view(1, -1)

    if eval:
      for k in self.metrics_eval.keys():
        self.metrics_eval[k].update(output, target)
    else:
      for k in self.metrics_train.keys():
        self.metrics_train[k].update(output, target)

  def compute_metrics(self, eval=True):
    results = {}
    if eval:
      for k in self.metrics_eval.keys():
        results[k] = self.metrics_eval[k].compute()
    else:
      for k in self.metrics_train.keys():
        results[k] = self.metrics_train[k].compute()

    return results

  def metric_keys(self, eval=True):
    if eval:
      return self.metrics_eval.keys()
    else:
      return self.metric_train.keys()

  def set_metric_transform(self, output_transform=None, target_transform=None):
    self.metrics_output_transform = output_transform
    self.metrics_target_transform = target_transform
