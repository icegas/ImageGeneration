from torch import nn
import torch

__all__ = ['UNET']

class UNETBlock(nn.Module):

    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1,
                  padding=1, activation=None, normalize=True, up_or_down=None):
        super(UNETBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

        self.up_or_down = None
        if up_or_down == "max":
            self.up_or_down = nn.MaxPool2d(2)
        elif up_or_down == "up":
            self.up_or_down = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        out = self.up_or_down(out) if self.up_or_down else out
        return out

def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

class UNET(nn.Module):

    def __init__(self, cfg) -> None:
        super(UNET, self).__init__()
        model_params = cfg.model.params
        # Sinusoidal embedding
        self.time_embed = nn.Embedding(model_params.n_steps, model_params.time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(model_params.n_steps, model_params.time_emb_dim).to(cfg.loader.device)
        self.time_embed.requires_grad_(False)

        # First half
        #self.te1 = self._make_te(cfg.time_emb_dim, 1)
        self.te, self.encoder, self.decoder = [], [], []
        width, height = cfg.loader.img_shape

        dec_in_c, dec_out_c = model_params.out_c[::-1], model_params.in_c[::-1]
        dec_out_c[-1] = model_params.decoder_out

        self.network_depth = len(model_params.in_c)
        for i in range(self.network_depth):
            self.te.append(self._make_te(model_params.time_emb_dim, model_params.in_c[i]).to(cfg.loader.device))
            self.encoder.append(nn.Sequential( 
                UNETBlock((model_params.in_c[i], width // 2**i, height // 2**i),
                 model_params.in_c[i], model_params.out_c[i], normalize=True, up_or_down='none').to(cfg.loader.device),
                UNETBlock((model_params.out_c[i], width // 2**i, height // 2**i),
                 model_params.out_c[i], model_params.out_c[i], normalize=True, up_or_down='max').to(cfg.loader.device),
                   
                   ) )

            self.decoder.append(nn.Sequential(
                UNETBlock((dec_in_c[i], width // 2**(self.network_depth - i), height // 2**(self.network_depth - i)),
                 dec_in_c[i], dec_out_c[i], normalize=True, up_or_down='none').to(cfg.loader.device),
                UNETBlock((dec_out_c[i], width // 2**(self.network_depth - i), height // 2**(self.network_depth - i)),
                 dec_out_c[i], dec_out_c[i], normalize=True, up_or_down='up').to(cfg.loader.device)
                 ))
        
        self.te_out = self._make_te(model_params.time_emb_dim, dec_out_c[i])
        self.b_out = nn.Sequential(
                UNETBlock((dec_out_c[-1], width , height),
                 dec_out_c[-1], dec_out_c[-1], normalize=True, up_or_down='none').to(cfg.loader.device),
                UNETBlock((dec_out_c[-1], width , height ),
                 dec_out_c[-1], dec_out_c[-1], normalize=False, up_or_down='none').to(cfg.loader.device)
                 )
        
        self.te = nn.ModuleList(self.te)
        self.encoder = nn.ModuleList(self.encoder)
        self.decoder = nn.ModuleList(self.decoder)
        self.conv_out = nn.Conv2d(dec_out_c[-1], model_params.in_c[0], 3, 1, 1).to(cfg.loader.device)
    
    def forward(self, x, t):
        t = self.time_embed(t)
        n = len(x)
        encoder_outs = []

        for i in range(self.network_depth):
            x = self.encoder[i](x + self.te[i](t).reshape(n, -1, 1, 1))
            encoder_outs.append(x)
        
        for i in range(self.network_depth):
            out = self.decoder[i](encoder_outs[-(i+1)])
            if i < self.network_depth - 1:
                out = out + encoder_outs[-(i+2)]
        
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)
        return out

    
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )