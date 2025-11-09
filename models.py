"""
Model architectures for CIFAR experiments
Includes ResNets, WideResNet, and Vision Transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# ResNet for CIFAR (32x32 images)
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """Bottleneck block for deeper ResNets"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_CIFAR(nn.Module):
    """ResNet for CIFAR (32x32 images)"""

    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(num_classes=10):
    """ResNet-20 for CIFAR"""
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet32(num_classes=10):
    """ResNet-32 for CIFAR"""
    return ResNet_CIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes)


def resnet44(num_classes=10):
    """ResNet-44 for CIFAR"""
    return ResNet_CIFAR(BasicBlock, [7, 7, 7], num_classes=num_classes)


def resnet56(num_classes=10):
    """ResNet-56 for CIFAR"""
    return ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes)


# ============================================================================
# Standard ResNet (for comparison or larger images)
# ============================================================================

def resnet18(num_classes=10):
    """Standard ResNet-18, adapted for CIFAR"""
    from torchvision.models import resnet18 as tv_resnet18
    model = tv_resnet18(num_classes=num_classes)
    # Modify first conv layer for CIFAR (32x32 instead of 224x224)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for CIFAR
    return model


def resnet34(num_classes=10):
    """Standard ResNet-34, adapted for CIFAR"""
    from torchvision.models import resnet34 as tv_resnet34
    model = tv_resnet34(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def resnet50(num_classes=10):
    """Standard ResNet-50, adapted for CIFAR"""
    from torchvision.models import resnet50 as tv_resnet50
    model = tv_resnet50(num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ============================================================================
# WideResNet
# ============================================================================

class WideBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def wrn28_10(num_classes=10, dropout=0.3):
    """WideResNet-28-10"""
    return WideResNet(depth=28, widen_factor=10, dropout_rate=dropout,
                      num_classes=num_classes)


# ============================================================================
# Vision Transformer (Simplified for CIFAR)
# ============================================================================

def vit_tiny(num_classes=10):
    """Vision Transformer Tiny for CIFAR"""
    try:
        from timm import create_model
        model = create_model(
            'vit_tiny_patch16_224',
            pretrained=False,
            num_classes=num_classes,
            img_size=32,
            patch_size=4,  # Smaller patches for CIFAR
        )
        return model
    except ImportError:
        raise ImportError("timm library required for ViT. Install with: pip install timm")


# ============================================================================
# Model Factory
# ============================================================================

def get_model(model_name, num_classes=10):
    """
    Factory function to get model by name

    Args:
        model_name: str, name of model architecture
        num_classes: int, number of output classes

    Returns:
        PyTorch model
    """
    models = {
        'resnet18': resnet18,
        'resnet20': resnet20,
        'resnet32': resnet32,
        'resnet34': resnet34,
        'resnet44': resnet44,
        'resnet50': resnet50,
        'resnet56': resnet56,
        'wrn28_10': wrn28_10,
        'vit_tiny': vit_tiny,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name](num_classes=num_classes)


if __name__ == '__main__':
    # Test models
    for model_name in ['resnet32', 'wrn28_10']:
        print(f"\nTesting {model_name}...")
        model = get_model(model_name, num_classes=100)
        x = torch.randn(2, 3, 32, 32)
        y = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {y.shape}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")