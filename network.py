import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class TCNLayer(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, k_size, first):
        super(TCNLayer,self).__init__()
        self.in_channels = in_channels
        self.first = first
        dil_pad = torch.tensor((dilation * (k_size - 1))/2)
        self.dil_conv = nn.Conv1d(in_channels, out_channels, k_size, padding = int(dil_pad), dilation = dilation)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.inplace = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        a = self.dil_conv(x)
        if self.first:
            zero = torch.zeros(1,self.in_channels,1).cuda()
            a = torch.cat((a,zero),dim=2)
        t = self.tanh(a)
        s = self.sigmoid(a)
        a = t*s
        skip = self.inplace(a)
        resid = skip + x
        return resid, skip

class TCNNetwork(nn.Module):
    #need a starting conv to get the input up to mid_channels
    def __init__(self,in_channels, mid_channels, out_channels, n_layers, k_size):
        super(TCNNetwork,self).__init__()
        self.layers = nn.ModuleList()
        first = True
        for i in range(n_layers):
            self.layers.append(TCNLayer(mid_channels,
                                        mid_channels,
                                        dilation = 2**i,
                                        k_size = k_size,
                                        first = first))
            first = False
        self.input = nn.Conv1d(in_channels, mid_channels, kernel_size=1)

        self.skip_to_out = nn.Sequential(nn.ReLU(),
                                         nn.ConvTranspose1d(mid_channels, mid_channels//2, kernel_size=4, stride=2, padding=1),
                                         nn.ReLU(),
                                         nn.ConvTranspose1d(mid_channels//2, out_channels, kernel_size=1),
                                         nn.Tanh())

        self.top_to_out = nn.Sequential(nn.ReLU(),
                                        nn.Conv1d(mid_channels, mid_channels, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Conv1d(mid_channels, out_channels, kernel_size=1),
                                        nn.Tanh())

    def forward(self,x):
        skips = []
        x = self.input(x)
        for idx, layer in enumerate(self.layers):
            x, skip = layer(x)
            skips.append(skip)
        #skip_stack = torch.cat(skips, dim=0)
        #skip_sum = torch.sum(skip_stack, dim=0).unsqueeze(0)
        #out = self.skip_to_out(skip_sum)
        out = self.top_to_out(x)
        return out
