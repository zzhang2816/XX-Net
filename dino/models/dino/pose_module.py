import torch.nn as nn

class BasicConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
        relu=nn.LeakyReLU, batch_norm=True, dropout=False, dropout_p=0.5
    ):
        super(BasicConv2d, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(relu())
        if dropout:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class BasicMLP(nn.Module):
    def __init__(self, in_channels, out_channels, 
                    relu=nn.LeakyReLU, dropout=False, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(in_channels,in_channels),
                relu(),
                nn.Linear(in_channels,in_channels), 
                relu(),
                nn.Linear(in_channels,out_channels))

    def forward(self, x):
        return self.net(x)

class PoseNet(nn.Module):
    def __init__(self, indim, hid_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.AvgPool2d(3, stride=2),
            BasicConv2d(indim, hid_dim, kernel_size=3, stride=1, padding=1),
            BasicConv2d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1),
            BasicConv2d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            BasicMLP(hid_dim,2)
        )
    
    def forward(self, x):
        res = self.net(x)
        return res[:,0],res[:,1]
        
def build_pose_module(in_dim,hid_dim):
    return PoseNet(in_dim,hid_dim)
        


