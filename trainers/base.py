import torch
from torch import optim
from tqdm import tqdm

import losses 
from dataloader import get_dataloader
import metrics
import models

class Base():

    def __init__(self, cfg, logger) -> None:
        super().__init__()
        self.logger = logger
        self.cfg = cfg
        self.epochs = cfg.model.epochs
        self.calc_metrics_on_train = cfg.model.calc_metrics_on_train
        self.calc_loss_on_eval = cfg.model.calc_loss_on_eval

        self.model = getattr(models, cfg.model.name)(cfg).to(cfg.loader.device)
        self.loss = getattr(losses, cfg.model.loss.name)()(cfg.model.loss.params)
        self.loader = get_dataloader(cfg)
        self.metrics = {metric_name : getattr(metrics, metric_name) for metric_name in cfg.metrics}

        self.optimizer = getattr(optim, self.cfg.optimizer.name)(
            self.model.parameters(), **self.cfg.optimizer.params)
        self.device = cfg.loader.device
        print("Number of model parameters: {}".format( sum([p.numel() for p in self.model.parameters()]) ) )
        print("Number of network parameters: {}".format( sum([p.numel() for p in self.model.network.parameters()]) ) )
    
    def init_metrics_and_losses(self, prefix='train_'):
        epoch_metrics = {prefix + metric_name : 0 for metric_name in self.cfg.model.metrics}
        epoch_losses =  {prefix + loss_name : 0 for loss_name in self.loss.names}
        
        return epoch_metrics, epoch_losses
    
    def add_loss_and_metrics(self, epoch_losses, epoch_metrics, loss, y_true, y_pred, eval=False, prefix='train_'):
        if loss:
            epoch_losses = {prefix+loss_name: epoch_losses[prefix+loss_name] + loss_value.item()
                        for loss_name, loss_value in loss.items()}

        if self.calc_metrics_on_train or eval:
            epoch_metrics = {prefix+metric_name : epoch_metrics[prefix+metric_name] + metric(y_true, y_pred) 
                             for metric_name, metric in self.metrics.items()}
        
        return epoch_metrics, epoch_losses
    
    def return_loss_and_metrics(self, epoch_losses, epoch_metrics, step, eval=False):
        epoch_losses = {loss_name: loss_value / step 
                        for loss_name, loss_value in epoch_losses.items() if loss_value != 0}

        if self.calc_metrics_on_train or eval:
            epoch_metrics = {metric_name : metric / step
                             for metric_name, metric in epoch_metrics.items()}
        
        return epoch_metrics, epoch_losses

    def train(self, epoch):

        epoch_metrics, epoch_losses = self.init_metrics_and_losses(prefix='train_')

        for step, batch in enumerate(tqdm(self.loader, 
                        leave=False, desc=f"Epoch {epoch + 1}/{self.epochs}", colour="#005500")):
            X, y_true = batch
            X = [X[i].to(self.device) for i in range(len(X))]
            y_true = y_true.to(self.device)

            y_pred = self.model(X)
            loss = self.loss(y_true, y_pred)
            self.optimizer.zero_grad()
            loss['main_loss'].backward()
            self.optimizer.step()
            epoch_metrics, epoch_losses = self.add_loss_and_metrics(epoch_losses, epoch_metrics,
                                                                    loss, y_true, y_pred, prefix='train_')
        
        return self.return_loss_and_metrics(epoch_losses, epoch_metrics, step)

    @torch.no_grad()
    def evaluate(self, epoch):
        epoch_metrics, epoch_losses = self.init_metrics_and_losses(prefix='val_')
        loss = None

        for step, batch in enumerate(tqdm(self.loader, 
                        leave=False, desc=f"Evaluation of {epoch + 1}/{self.epochs}", colour="#005500")):
            X, y_true = batch

            y_pred = self(X)
            if self.calc_loss_on_eval:
                loss = self.loss(y_true, y_pred)
            epoch_metrics, epoch_losses = self.add_loss_and_metrics(loss, y_true, y_pred, eval=True, prefix='val_')
        
        return self.return_loss_and_metrics(epoch_losses, epoch_metrics, step, eval=True)
    
    def save(self, model_savepath, e):
        return self.model.save(model_savepath, e)