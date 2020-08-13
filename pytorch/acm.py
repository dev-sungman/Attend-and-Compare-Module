import torch
import torch.nn as nn
import torch.nn.functional as F

class ACMBlock(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(ACModule, self).__init__()

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
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),
            nn.ReLU(),
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),
            nn.Sigmoid()
        )

    def _normalize_features(self, x):
        # get mean vecotr by axis C
        # input : [B, C, H, W]
        # output : [B, C, 1, 1]

        c_mean = torch.mean(x, 1) 
        return c_mean


    def forward(self, x):
        mean_features = self._normalize_features(x)
        normalized_features = x - mean_features

        K = self.k_conv(x)
        Q = self.q_conv(x)
        
        K = torch.matmul(K, x)
        Q = torch.matmul(Q, x)
        
        diff = K-Q
        channel_weight = self.global_pooling(x)

        return torch.matmul(channel_weight, (x+(K-Q)))



if __name__ == '__main__':
    test_vector = torch.randn((1,2,5,5), dtype=torch.float32)
    acm = ACMBlock(in_channels=test_vector.shape[1])

    result = acm(test_vector)
    print(result)
