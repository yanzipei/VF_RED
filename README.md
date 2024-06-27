# Generalized Robust Fundus Photography-based Vision Loss Estimation for High Myopia
This paper is accepted by MICCAI 2024.

## Implementation

### Feature Extraction
Take the pretrained resnet18 as an example, we extract features from fundus photo.
```python
import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms

# define forward for resnet
def resnet_features(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    return x

# load pretrained model
backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
backbone.eval()

# defined image transform
input_resolution = 384
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = transforms.Compose([transforms.Resize(size=input_resolution),
                                transforms.CenterCrop(size=input_resolution),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=mean, std=std)])

# extract features
image_list = load_your_fundus_images()

feat_list = []
for img in image_list:
    img = transform(img)
    feat = resnet_features(backbone, img)
    feat_list.append(feat_list)
```


### Model
The model is the basic MLP, which is defined as follows:
```python
from typing import List
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, dim_list: List[int], act_func, bias: bool = True):
        super(MLP, self).__init__()
        assert len(dim_list) >= 2

        encoder = []

        if len(dim_list) == 2:
            self.encoder = nn.Identity()
        else:
            for i in range(len(dim_list) - 2):
                print(i, dim_list[i], dim_list[i + 1])
                encoder += [nn.Linear(dim_list[i], dim_list[i + 1], bias=bias), act_func]
            self.encoder = nn.Sequential(*encoder)

        self.regressor = nn.Linear(dim_list[-2], dim_list[-1], bias=bias)

    def forward(self, x, return_feat=False):
        feat = self.encoder(x)
        x = self.regressor(feat)

        if return_feat:
            return x, feat
        else:
            return x

    def feature(self, x):
        return self.encoder(x)
```
For the model that takes the extracted features from pretrained resnet-18, it is defined as follows:
```python
model = MLP(dim_list=[512, 512, 52], act_func=nn.ReLU(inplace=True), bias=True)
```

### Training model with MC-SURE
```python
import torch

# define MC-SURE
def mc_sure(z: torch.Tensor, model, sigma: torch.Tensor, eps: float):
    """
    MC-SURE for batch.
    :param z: feature tensor, shape: [N, K]
    :param model: the denoising model
    :param sigma: sigma vector, shape: [N], sigma for each feature
    :param eps: epsilon, float.
    :return: sure loss vector, shape: [N].
    """
    assert z.ndim == 2  # [N, K]
    assert sigma.ndim == 2  # [N, 1]
    assert z.shape[0] == sigma.shape[0]

    K = z.shape[1]
    var = sigma ** 2  # [N, 1]
    output = model(z)

    b = torch.randn(z.shape, device=z.device)

    z_hat = z + b * eps
    output_hat = model(z_hat)

    loss = ((z - output) ** 2).mean(dim=1) - var + 2 * var * (b * (output_hat - output)).sum(dim=1) / (K * eps)  # [N]

    return loss

lr = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

eps = 1e-5
lam = 1.0
model.train()

# training
for feat, vf, sigma in dataloader:
    # feat: [N, K], batch of extracted feature vector
    # vf: [N, M], batch of target VF vector
    # sigma: [N, 1], batch of sigma scalar, which is estimated on feat.

    pred_vf, denoised_feat = model(feat, True)

    pred_loss = ((pred_vf - vf) ** 2).mean(1)  # [N]
    sure_loss = mc_sure(feat, model.encoder, sigma, eps)  # [N]

    loss = pred_loss + lam * sure_loss
    loss = loss.mean()

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Repos for baselines

EMD & SOFT & OLL: https://github.com/glanceable-io/ordinal-log-loss

CORAL: https://github.com/Raschka-research-group/coral-cnn


VF-HM: https://github.com/yanzipei/VF-HM

OE: https://github.com/needylove/OrdinalEntropy


## Citation
If this work is useful for your research, please kindly cite it:
```
@inproceedings{yan2024vfred,
title={Generalized robust fundus photography-based vision loss estimation for high myopia},
author={Yan, Zipei and Liang, Zhile and Liu, Zhengji and Wang, Shuai and Chun, Rachel and Li, Jizhou and Kee, Chea-su and Liang, Dong},
booktitle={MICCAI},
year={2024},
}
```
