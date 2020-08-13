import torch
import torch.nn as nn

class ACModule(nn.Module):
    def __init__(self, in_channels, groups=2):
        super(ACModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.groups = groups
        

    def _normalize_features(self, x):
        # get mean vecotr by axis C
        # x : [B, C, H, W]
        c_mean = torch.mean(x, 1) # [B, C]
        return x - c_mean

    def _select_features(self):
        print('TODO')
        


    def forward(self, x):
        mean_vector = self._normalize_features(x)
        print(x)
        print(mean_vector)












if __name__ == '__main__':
    test_vector = torch.randn((1,3,5,5), dtype=torch.float32, device='cuda:0')
    acm = ACModule(in_channels=3)

    mean_test_vector = acm(test_vector)
