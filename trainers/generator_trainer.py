from .base import Base

import torch
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from pathlib import Path
from uuid import uuid4
import numpy as np
import shutil

__all__=['GeneratorTrainer']

class GeneratorTrainer(Base):

    def __init__(self, cfg, logger) -> None:
        super().__init__(cfg, logger)
        self.eval_batches = cfg.loader.eval_batches
        self.channels = cfg.loader.img_channels
        self.height, self.width = cfg.loader.img_shape
        self.gen_img_savepath = Path(cfg.loader.generated_images_savepath)
        self.nrow = round(np.sqrt(self.loader.batch_size))
        self.save_each_epoch = cfg.logger.save_each_epoch
    
    @torch.no_grad()
    def evaluate(self, epoch):
        epoch_metrics, epoch_losses = self.init_metrics_and_losses(prefix='val_')
        loss, y_true = None, None
        if self.gen_img_savepath.exists():
            shutil.rmtree(str(self.gen_img_savepath))
        self.gen_img_savepath.mkdir(exist_ok=True)

        if epoch % self.save_each_epoch == 0 and epoch != 0:
            for step in tqdm(range(self.eval_batches), 
                            leave=False, desc=f"Generation of {epoch + 1}/{self.epochs}", colour="#880000"):

                gen_images = self.model.generate_new_images(self.loader.batch_size, device=self.device,
                                c=self.channels, h=self.height, w=self.width).detach().cpu()

                epoch_metrics, epoch_losses = self.add_loss_and_metrics(epoch_losses, epoch_metrics, loss, y_true, gen_images, eval=True, prefix='val_')

                savename = self.gen_img_savepath / (str(uuid4()) + '.png')
                grid = make_grid(gen_images, self.nrow)
                save_image(grid, savename, format='png')
        
            self.logger.log_artifact(str(self.gen_img_savepath), 'gen_images_epoch_{}'.format(epoch))
        else:
            step = 1
        return self.return_loss_and_metrics(epoch_losses, epoch_metrics, step, eval=True)
    