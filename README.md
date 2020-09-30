# Second Order Optimization for Adversarial Robustness and Interpretability
Repository based on arXiv preprint https://arxiv.org/abs/2009.04923

## Prerequisites
* PyTorch 1.3.0
* NumPy 1.16.5
* TQDM
* [Robustness package](https://robustness.readthedocs.io/en/latest/index.html)


**Note**: For SVHN support you will need to use the robustness code from the forked [robustness repo](https://github.com/Jay-Roberts/robustness). Clone the repo into your scorpio directory and change `import robustness.<module>` to `import robustness.robustness.<module>`. You can then use SVHN models as:

```python
from robustness.robustness.model_utils import make_and_restore_model
from robustness.robustness.datasets import SVHN

ds = SVHN('path/to/SVHN/data')

model, _ = make_and_restore_model(arch='resnet18', dataset=ds, resume_path='path/to/svhn_Linf_FE_N3.pt')
model.cuda()
model.eval()
```

## Pre-trained models
All pre-trained models are based on the ResNet-18 architecture
* [CIFAR10 L2](https://www.dropbox.com/s/htvc5hjwcft2mj1/cifar_L2_FE_N3.pt?dl=0)
* [CIFAR10 Linf](https://www.dropbox.com/s/3ph8w74ke57kb9w/cifar_Linf_FE_N3.pt?dl=0)
* [SVHN L2](https://www.dropbox.com/s/05d8o7652vnjkq5/svhn_L2_FE_N3.pt?dl=0)
* [SVHN Linf](https://www.dropbox.com/s/5aikbe2o4pjcwnd/svhn_Linf_FE_N3.pt?dl=0)

## Sample code
A sample notebook is provided for loading and evaluating a pre-trained model
```
scorpio_cifar10_test.ipynb
```
