import torch
from torch import nn

def Conv(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
    return nn.Sequential(
        nn.Conv2d(
            n_input, n_output,
            kernel_size=k_size,
            stride=stride,
            padding=padding, bias=False),
        nn.BatchNorm2d(n_output),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(p=0.2, inplace=False))

# This is tiny-yolo
class YOLO(torch.nn.Module):
    def __init__(self, nc=32, S=13, BOX=5, CLS=2):
        super(YOLO, self).__init__()
        self.nc = nc
        self.S = S
        self.BOX = BOX
        self.CLS = CLS
        self.net = nn.Sequential(
            nn.Conv2d(
                3, nc,
                kernel_size=4,
                stride=2,
                padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            Conv(nc, nc, 3,2,1),
            Conv(nc, nc*2, 3,2,1),
            Conv(nc*2, nc*4, 3,2,1),
            Conv(nc*4, nc*8, 3,2,1),
            Conv(nc*8, nc*16, 3,1,1),
            Conv(nc*16, nc*8, 3,1,1),
            Conv(nc*8, BOX*(4+1+CLS), 3,1,1),
        )
        
    def forward(self, input):
        output_tensor = self.net(input)
        output_tensor = output_tensor.permute(0, 2,3,1)
        W_grid, H_grid = self.S, self.S
        output_tensor = output_tensor.view(-1, H_grid, W_grid, self.BOX, 4+1+self.CLS)
        return output_tensor