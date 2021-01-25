import torch
import torch.nn as nn
import torch.nn.functional as F

class ACMBlock(nn.Module):
    def __init__(self, in_channels, groups=1):
        super(ACMBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.groups = groups
        self.k_conv = nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),

        self.q_conv = nn.Conv2d(self.in_channels, self.out_channels, (1,1), groups=self.groups),

        self.global_pooling = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels//2, (1,1), groups=self.groups),
            nn.ReLU(),
            nn.Conv2d(self.out_channels//2, self.out_channels, (1,1), groups=self.groups),
            nn.Sigmoid()
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.normalize = nn.Softmax(dim=3)

    def _get_normalized_features(self, x):
        ''' Get mean vector by channel axis '''
        
        c_mean = self.avgpool(x)
        return c_mean

    def _get_orth_loss(self, K, Q):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        orth_loss = cos(K, Q)
        orth_loss = torch.mean(orth_loss, dim=0)
        return orth_loss

    def forward(self, x):
        # get normalized features
        mean_features = self._get_normalized_features(x)
        normalized_features = x - mean_features
        
        # get features
        K = self.k_conv(normalized_features)
        b, c, h, w = K.shape
        K = K.view(b, c, 1, h*w)
        K = self.normalize(K)

        Q = self.q_conv(normalized_features)
        b, c, h, w = Q.shape
        Q = Q.view(b, c, 1, h*w)
        Q = self.normalize(Q)
        
        K = torch.einsum('bchw,bchw->nc',[K, x]).unsqueeze(-1).unsqueeze(-1)
        Q = torch.einsum('bchw,bchw->nc',[Q, x]).unsqueeze(-1).unsqueeze(-1)
        
        # global information
        channel_weights = self.global_pooling(mean_features)

        out = channel_weights*(x + (K-Q))

        # orthogonal loss
        orth_loss = self._get_orth_loss(K, Q)

        return out, orth_loss
    

### Test
if __name__ == '__main__':
    test_vector = torch.randn((1,2,3,3), dtype=torch.float32)
    acm = ACMBlock(in_channels=test_vector.shape[1])

    result = acm(test_vector)
