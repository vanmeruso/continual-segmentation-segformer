import torch
import torch.nn as nn
import torch.nn.functional as F 
from .segformer_head import SegFormerHead
from . import mix_transformer
from torchsummary import summary

class WeTr(nn.Module):
    def __init__(self, backbone, num_classes=20, embedding_dim=256, pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        #self.in_channels = [32, 64, 160, 256]
        self.in_channels = [64, 128, 320, 512]

        self.encoder = getattr(mix_transformer, backbone)()
        #self.in_channels = self.encoder.embed_dims
        ## initilize encoder
        if pretrained:
            state_dict = torch.load('pretrained/'+backbone+'.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict,)

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.num_classes)
        
        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes, kernel_size=1, bias=False)

    def _forward_cam(self, x):
        
        cam = F.conv2d(x, self.classifier.weight)
        cam = F.relu(cam)
        
        return cam

    def get_param_groups(self):

        param_groups = [[], [], []] # 
        
        for name, param in list(self.encoder.named_parameters()):
            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        for param in list(self.decoder.parameters()):

            param_groups[2].append(param)
        
        param_groups[2].append(self.classifier.weight)

        return param_groups

    def forward(self, x):

        _x = self.encoder(x)
        _x1, _x2, _x3, _x4 = _x
        cls = self.classifier(_x4)

        return self.decoder(_x)

'''
if __name__ == '__main__':
    model = WeTr(backbone='mit_b2', num_classes=21,
                embedding_dim=256,
                pretrained=True)
    
    img = torch.rand((1, 3, 512, 512))

    # summary(model, input_size=(3, 512, 512), device='cpu')
    # Encoding OUTPUT Shape: torch.Size([1, 64, 128, 128]), torch.Size([1, 128, 64, 64]), torch.Size([1, 320, 32, 32]), torch.Size([1, 512, 16, 16])
    # print(f'OUTPUT Shape: {model(img)[0].shape}, {model(img)[1].shape}, {model(img)[2].shape}, {model(img)[3].shape}')
    
    # Decoding OUTPUT Shape: torch.Size([1, 21, 128, 128])
    print(f'OUTPUT Shape: {model(img).shape}')
'''