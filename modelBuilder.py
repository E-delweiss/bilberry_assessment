from configparser import ConfigParser

import torch
import torchvision
from torchinfo import summary
from icecream import ic

class ResNetBilberry(torch.nn.Module):
    def __init__(self, pretrained):
        super(ResNetBilberry, self).__init__()
        if pretrained:
            PT_weights = torchvision.models.ResNet34_Weights.DEFAULT
        else:
            PT_weights = None

        ### Load ResNet model
        self.resnet = torchvision.models.resnet34(weights=PT_weights)

        ### Freeze ResNet weights
        it = 0
        if pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False
                if it > 100:
                    break
                it += 1

        ## Head part
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            # torch.nn.Dropout(p=0.5),
            torch.nn.Linear(num_ftrs, num_ftrs//2),
            torch.nn.LayerNorm(num_ftrs//2),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, input:torch.Tensor)->tuple:
        """
        TODO
        """
        x = self.resnet(input)
        x = torch.nn.Sigmoid()(x)
        return x


class EfficientNetBilberry(torch.nn.Module):
    def __init__(self, pretrained):
        super(EfficientNetBilberry, self).__init__()
        if pretrained:
            PT_weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        else:
            PT_weights = None

        ### Load ResNet model
        self.efficientnet = torchvision.models.efficientnet_b0(weights=PT_weights)
        
        ### Freeze ResNet weights
        if pretrained:
            for param in self.efficientnet.parameters():
                param.requires_grad = False

        ### Head part
        num_ftrs = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def forward(self, input:torch.Tensor)->tuple:
        """
        TODO
        """
        x = self.efficientnet(input)
        x = torch.nn.Sigmoid()(x)
        return x


def resNetBilberry(load_resNetBilberry_weights:bool=False, pretrained:bool=True) -> ResNetBilberry:
    """
    Load ResNet18 model from torchvision

    Args:
        load_resNetBilberry_weights (bool, optional): _description_. Defaults to False.
        pretrained (bool, optional): _description_. Defaults to True.

    Returns:
        ResNetBilberry: _description_
    """

    config = ConfigParser()
    config.read("config.ini")

    model = ResNetBilberry(pretrained)

    if load_resNetBilberry_weights:
        resNetBilberry_weights = config.get("WEIGHTS", "resNetBilberry_weights")
        print(f"Loading {resNetBilberry_weights} weights...")
        model.load_state_dict(torch.load(resNetBilberry_weights))
    return model

def efficientNetBilberry(load_efficientNetBilberry:bool=False, pretrained:bool=True) -> EfficientNetBilberry:
    """
    Load EfficientNetB0 model from torchvision.

    Args:
        load_efficientNetBilberry (bool, optional): load finetuned weights. Defaults to False.
        pretrained (bool, optional): load pretrained model on ImageNet. Defaults to True.

    Returns:
        EfficientNetBilberry: _description_
    """

    config = ConfigParser()
    config.read("config.ini")

    model = EfficientNetBilberry(pretrained)
    if load_efficientNetBilberry:
        efficientNetBilberry_weights = config.get("WEIGHTS", "efficientNetBilberry_weights")
        model.load_state_dict(torch.load(efficientNetBilberry_weights))
    return model


if __name__ == "__main__":
    model = ResNetBilberry(pretrained=True)

    BATCH_SIZE = 64
    img_test = torch.rand(BATCH_SIZE, 3, 140, 140)
    summary(model, input_size = img_test.shape)
    print(type(model))
