import torch.nn as nn
import math
import torch
class OldNNDenoiser(nn.Module):
    def __init__(self, img_size=128, channels=3, hidden=2048):
        super().__init__()
        input_dim = img_size * img_size * channels

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//2, hidden//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//4, hidden//8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//8, hidden//4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//4, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden//2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, input_dim),
            nn.Tanh()  
        )
        self.img_size = img_size
        self.channels = channels

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, -1)
        x = self.net(x)
        x = x.view(B, self.channels, self.img_size, self.img_size)
        return x
    

class NNDenoiser(nn.Module):
    def __init__(self, img_size=128, channels=3):
        super().__init__()
        self.input_dim = img_size * img_size * channels
        hidden = 4096
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.input_dim)
        )
        # self.time_embed = nn.Sequential(nn.Linear(hidden, hidden),
        #                                 nn.ReLU(),
        #                                 nn.Linear(hidden,self.input_dim))
        
        self.img_size = img_size
        self.channels = channels

    def gen_t_embedding(self, t, max_positions=10000):
        t = t * max_positions
        half_dim = self.input_dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.input_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb

    def forward(self, x,t):
        B = x.shape[0]
        x = x.view(B, -1)
        t = self.gen_t_embedding(t.view(B, -1))
        x = x + t.view(B, -1)
        x = self.net(x)
        x = x.view(B, self.channels, self.img_size, self.img_size)
        return x