import torch.nn as nn
import torch
import torch.nn.functional as F
from basicsr.models.archs.layers.transformer import Encoder, Decoder

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
     
class Conv1x1Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_1x1_convs):
        super(Conv1x1Decoder, self).__init__()
        self.gelu = GELU()
        self._1x1_convs = nn.ModuleList([])

        for _ in range(num_1x1_convs):
            self._1x1_convs.append(nn.ModuleList([
                nn.Conv2d(in_dim, in_dim, 1, 1, bias=False),
                self.gelu,
            ]))

        self.out_layer = nn.Conv2d(in_dim, out_dim, 3, 1, 1, bias=False)

    def forward(self, x):

        for (_1x1_conv, nonlinearity) in self._1x1_convs:
            x = _1x1_conv(x)
            x = nonlinearity(x)

        out = self.out_layer(x)

        return out

class Unet(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4], z_channels=256):
        super(Unet, self).__init__()

        self.dim = dim

      
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        self.encoder = Encoder(dim=dim, level=level, num_blocks=num_blocks, use_feature=True)
        self.decoder = Decoder(dim=dim, level=level, num_blocks=num_blocks)

        self.fcomb = Conv1x1Decoder(self.dim + z_channels, out_dim, num_1x1_convs=3)

    def rsample(self, mu, var):
        '''
        Return gaussian sample of (mu, var) using reparameterization trick.
        '''
        eps = torch.randn_like(mu)
        z = mu + eps*torch.sqrt(var)
        return z 
    
    def sampling(self, x, pm, pv, num_samples):
        skip_connect = x
        fea = self.embedding(x)
        fea, fea_encoder = self.encoder(fea)
        fea = self.decoder(fea, fea_encoder)

        outputs, darkness = [], []
        for _ in range(num_samples):
            z = self.rsample(pm, pv)
            broadcast_z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), fea.size(2), fea.size(3))
            output = self.fcomb(torch.cat([fea, broadcast_z], dim=1))
            enhance = skip_connect + output
            darkness.append(output)
            outputs.append(enhance)

        return outputs , darkness
    
    def interpolate(self, x, pm, pv, num_samples):

        skip_connect = x
        fea = self.embedding(x)
        fea, fea_encoder = self.encoder(fea)
        fea = self.decoder(fea, fea_encoder)

        z_start = self.rsample(pm, pv)
        z_end = self.rsample(pm, pv)

        outputs = []

        for i in range(num_samples):
   
            alpha = i / (num_samples - 1)
            z_interp = (1 - alpha) * z_start + alpha * z_end

            broadcast_z = z_interp.view(z_interp.size(0), z_interp.size(1), 1, 1).expand(
                z_interp.size(0), z_interp.size(1), fea.size(2), fea.size(3)
            )

            output = self.fcomb(torch.cat([fea, broadcast_z], dim=1))
            enhance = skip_connect + output
            outputs.append(enhance)
        return outputs 

    
    def forward(self, x, z):
        fea = self.embedding(x)

        fea, fea_encoder = self.encoder(fea)
        fea = self.decoder(fea, fea_encoder)
        
        broadcast_z = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), fea.size(2), fea.size(3))
        output = self.fcomb(torch.cat([fea, broadcast_z], dim=1))
        
        return output

if __name__ == '__main__':
    unet = Unet(in_dim=3, out_dim=3, dim=40, level=2, num_blocks=[2, 4, 2], z_channels=10,  use_add=True)
    # Test
    _input = torch.randn([8, 3, 128, 128])
    z = torch.randn([8,10])

    output = unet(_input, z)
    print(output.shape)