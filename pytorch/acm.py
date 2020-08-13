import torch
import torch.nn as nn
import torch.nn.functional as F

class ACModule(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(ACModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.groups = groups

    
        self.k_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),
            nn.Softmax()
        )

        self.q_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),
            nn.Softmax()
        )

        self.global_pooling = nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups)

    def _normalize_features(self, x):
        # get mean vecotr by axis C
        # input : [B, C, H, W]
        # output : [B, C, 1, 1]

        c_mean = torch.mean(x, 1) 
        return c_mean

    def _select_features(self):
        print('TODO')
        


    def forward(self, x):
        mean_features = self._normalize_features(x)
        normalized_features = x - mean_features
        

        K = self.k_conv(x)
        Q = self.q_conv(x)

        print(K)
        print(Q)













if __name__ == '__main__':
    test_vector = torch.randn((1,3,5,5), dtype=torch.float32)
    acm = ACModule(in_channels=3)

    mean_test_vector = acm(test_vector)
