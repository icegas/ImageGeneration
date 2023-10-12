from mlflow import log_metric, log_param, log_artifacts, log_artifact, log_params
import mlflow
import subprocess
import shutil
from pathlib import Path
from omegaconf.dictconfig import DictConfig

__all__ = ['Logger']

class Logger():
    def __init__(self, cfg) -> None:
        mlflow.set_tracking_uri(cfg.logger.tracking_uri) 
        mlflow.set_experiment(cfg.logger.experiment_name)
        mlflow.set_tag("mlflow.runName", cfg.logger.run_name)
        self.save_params(cfg)
        for path in cfg.logger.log_dirs:
            log_artifact(path)
    
    def save_params(self, cfg):
        for k, v in cfg.items():
            if isinstance(v, DictConfig):
               self.save_params(v) 
            elif isinstance(v, list):
                for i in range(len(v)):
                    log_param(k + '_' + str(i), v[i])
            else:
                if k != "name":
                    log_param(k, v)

    def end_run(self):
        mlflow.end_run()
    
    def log_artifact(self, artifact, name):
        log_artifact(artifact, name)

    def log_epoch(self, metrics, model_paths):

        for metric_name, metric in metrics.items():
            log_metric(metric_name, metric)
        
        for model_path in model_paths:
            log_artifact(model_path)