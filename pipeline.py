import importlib
import os
import shutil
from utils import Logger
import numpy as np
import uuid
import trainers

class Pipeline():

    def __init__(self, cfg) -> None:
        self.logger = Logger(cfg)
        self.model = getattr(trainers, cfg.loader.trainer)(cfg, self.logger)
                            
        #self.eval = cfg.model.eval
        self.epochs = cfg.model.epochs
        self.save_epoch_percent = cfg.model.save_epoch_percent
        self.start_epoch = cfg.model.start_epoch
        
        self.model_savepath = 'exps/' + str(uuid.uuid4()) + '/'
        os.makedirs(self.model_savepath, exist_ok=True)
        self.print_metrics = cfg.logger.print_metrics
        self.eval = cfg.model.eval
        self.save_each_epoch = cfg.logger.save_each_epoch
    
    def add_best_value(self, df : dict, func):
        best_metrics = {}
        for k, v in df.items():
            best_metrics[k +'_best'] = func(df[k])
        df.update(best_metrics)
        return df
    
    def merge_all_metrics(self, metrics):
        log_metrics = {}
        for m in metrics:
            log_metrics.update(m)
        
        return log_metrics
    
    def run(self):

        train_lds, val_lds, epochs = [], [], []
        for e in range(self.start_epoch, self.epochs):
            epochs.append(e)
            train_metrics, train_losses = self.model.train(e)
            train_metrics = self.add_best_value(train_metrics, np.max)
            train_losses =  self.add_best_value(train_losses, np.min)
            all_metrics = [train_metrics, train_losses]

            if self.eval:
                val_metrics, val_losses = self.model.evaluate(e)
                val_metrics = self.add_best_value(val_metrics, np.max)
                val_losses =  self.add_best_value(val_losses, np.min)
                all_metrics.append(val_metrics)
                all_metrics.append(val_losses)

            model_paths = []
            if e % self.save_each_epoch == 0:
                model_paths = self.model.save(self.model_savepath, e)
            
            log_metrics = self.merge_all_metrics(all_metrics)
            self.logger.log_epoch(log_metrics, model_paths)

            for m in self.print_metrics:
                print("{}: {:.4f}".format(m, log_metrics[m]))
        
        self.logger.end_run()