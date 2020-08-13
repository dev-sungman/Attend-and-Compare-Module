import torch
import torch.nn as nn
import torch.nn.functional as F

class ACMBlock(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(ACMBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.groups = groups

    
        self.k_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),
            nn.Softmax(dim=1)
        )

        self.q_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),
            nn.Softmax(dim=1)
        )

        self.global_pooling = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels//2, (1,1), groups=self.groups),
            nn.ReLU(),
            nn.Conv2d(self.out_channels//2, self.out_channels, (1,1), groups=self.groups),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def _get_normalized_features(self, x):
        '''
        Get mean vector by channel axis
        Args:
            input : tensor [B, C, H, W]
        
        Return:
            mean tensor [B, C, 1, 1]
        '''

        c_mean = self.avgpool(x)
        return c_mean


    def forward(self, x):
        # get normalized features
        mean_features = self._get_normalized_features(x)
        normalized_features = x - mean_features
        
        # get features
        K = self.k_conv(x)
        Q = self.q_conv(x)
        
        K = torch.einsum('nchw,nchw->nc',[K, x]).unsqueeze(-1).unsqueeze(-1)
        Q = torch.einsum('nchw,nchw->nc',[Q, x]).unsqueeze(-1).unsqueeze(-1)
        
        diff = K-Q
        
        # global information
        channel_weights = self.global_pooling(mean_features)

        return channel_weights * (x+(K-Q))


### Test
if __name__ == '__main__':
    test_vector = torch.randn((1,2,3,3), dtype=torch.float32)
    acm = ACMBlock(in_channels=test_vector.shape[1])

    result = acm(test_vector)
