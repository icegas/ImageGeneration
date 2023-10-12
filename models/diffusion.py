import torch
import numpy as np

import models.networks

__all__ = ['DiffusionModel']

class DiffusionModel(torch.nn.Module):

    def __init__(self, cfg) -> None:
        super(DiffusionModel, self).__init__()
        self.network = getattr(models.networks, cfg.model.network)(cfg).to(cfg.loader.device)
        params = cfg.model.params

        self.n_steps = params.n_steps
        self.betas = torch.linspace(params.min_beta, params.max_beta, params.n_steps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(cfg.loader.device)
    
    def forward(self, X):
        # Make input image more noisy (we can directly skip to the desired step)
        x0, t, eta = X
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        #if eta is None:
            #eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return self.network(noisy, t.reshape(n, -1))

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

    def save(self, model_savepath, e):
        savepath = "{}model_{}.pt".format(model_savepath , e)
        torch.save(self.state_dict(), savepath)
        return [savepath]

    def generate_new_images(self, n_samples=16, device='cuda', c=1, h=28, w=28):
        frames = []

        with torch.no_grad():

            # Starting from random noise
            x = torch.randn(n_samples, c, h, w).to(device)

            for idx, t in enumerate(list(range(self.n_steps))[::-1]):
                # Estimating noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
                eta_theta = self.backward(x, time_tensor)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(n_samples, c, h, w).to(device)

                    # Option 1: sigma_t squared = beta_t
                    beta_t = self.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Option 2: sigma_t squared = beta_tilda_t
                    # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                    # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                    # sigma_t = beta_tilda_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z

                if t == 0:
                    # Putting digits in range [0, 255]
                    for i in range(len(x)):
                        x[i] -= torch.min(x[i])
                        x[i] /= torch.max(x[i])

                    ## Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                    #frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                    #frame = frame.cpu().numpy().astype(np.uint8)

                    ## Rendering frame
                    #frames.append(frame)

        return x