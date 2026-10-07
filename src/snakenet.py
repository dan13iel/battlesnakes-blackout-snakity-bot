import numpy as np
import torch
import torch.nn as nn
import obsmem as om

PRIV_CH = 7 # full-board occupancy(0-3) + heads + food + offboard

# INV_ROT_ACTION[rot, canonical_engine_action] -> raw engine action (world frame)
# Only took a few hours to debug ;-;
INV_ROT_ACTION = np.array([[0, 1, 2, 3],
                           [3, 2, 0, 1],
                           [1, 0, 3, 2],
                           [2, 3, 1, 0]], dtype=np.uint8)


class SnakeNet(nn.Module):
    # the only reason this is somewhat trainable is because of the squeeze so dont remove that
    def __init__(self, in_ch=om.C, crop=om.CROP, n_scal=5, n_act=3, ch=(16, 32), squeeze=8, hid=192, mem=96, priv_ch=None):
        super().__init__()
        self.mem = mem
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, ch[0], 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch[1], squeeze, 1), nn.ReLU(), # channel squeeze before flatten
            nn.Flatten(),
        )
        self.fc = nn.Sequential(nn.Linear(squeeze * crop * crop + n_scal, hid), nn.ReLU())
        self.mem_cell = nn.LSTMCell(hid, mem)
        self.pi = nn.Linear(mem, n_act)
        if priv_ch is None:
            self.priv_enc = None
            self.v = nn.Linear(mem, 1)
        else:
            self.priv_enc = nn.Sequential(
                nn.Conv2d(priv_ch, ch[0], 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch[0], ch[1], 3, padding=1), nn.ReLU(),
                nn.Conv2d(ch[1], squeeze, 1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(squeeze * 15 * 15, 128), nn.ReLU(),
            )
            self.v = nn.Sequential(nn.Linear(mem + 128, 128), nn.ReLU(), nn.Linear(128, 1))
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, gain=0.01) # near-uniform initial policy
        out = self.v[-1] if self.priv_enc is not None else self.v
        nn.init.orthogonal_(out.weight, gain=1.0)

    def forward(self, x, scal, hc, priv=None):
        feat = self.fc(torch.cat((self.conv(x), scal), dim=1))
        h, c = self.mem_cell(feat, hc)
        logits = self.pi(h)
        if self.priv_enc is None:
            value = self.v(h).squeeze(-1)
        elif priv is not None:
            value = self.v(torch.cat((h, self.priv_enc(priv)), dim=1)).squeeze(-1)
        else:
            value = None
        return logits, value, (h, c)

    def forward_seq(self, x, scal, hc, priv, T, B):
        """forward() but its *vertorised*
        Its just a batched version of forward so its much cheaper to run when training. For inference it doesnt really matter."""
        
        feat = self.fc(torch.cat((self.conv(x), scal), dim=1)).view(T, B, -1)
        h, c = hc
        hs = []
        for t in range(T):
            h, c = self.mem_cell(feat[t], (h, c))
            hs.append(h)
        hs = torch.stack(hs).view(T * B, -1)
        logits = self.pi(hs).view(T, B, -1)
        if self.priv_enc is None:
            value = self.v(hs).view(T, B)
        else:
            value = self.v(torch.cat((hs, self.priv_enc(priv)), dim=1)).view(T, B)
        return logits, value, (h, c)