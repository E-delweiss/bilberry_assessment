from configparser import ConfigParser

import torch
import torchvision
from torchinfo import summary
from icecream import ic

class ResNetBilberry(torch.nn.Module):
    def __init__(self, pretrained):
        super(ResNetBilberry, self).__init__()
        ### Load ResNet model
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features

        ### Freeze ResNet weights
        if pretrained:
            for param in self.resnet.parameters():
                param.requires_grad = False

        ### Head part
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(num_ftrs, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, input:torch.Tensor)->tuple:
        """
        TODO
        """
        x = self.resnet(input)
        x = torch.nn.Sigmoid()(x)
        return x


def resNetBilberry(load_resNetBilberry_weights:bool=False, pretrained:bool=True) -> ResNetBilberry:
    """
    TODO

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
        model.load_state_dict(torch.load(resNetBilberry_weights))
    return model

if False:
    def efficientNetBilberry(load_efficientNetBilberry:bool=False, pretrained:bool=True) -> EfficientNetBilberry:
        """
        TODO

        Args:
            load_efficientNetBilberry (bool, optional): _description_. Defaults to False.
            pretrained (bool, optional): _description_. Defaults to True.

        Returns:
            EfficientNetBilberry: _description_
        """

        config = ConfigParser()
        config.read("config.ini")

        model = EfficientNetBilberry(pretrained)
        if load_yoloweights:
            efficientNetBilberry_weights = config.get("WEIGHTS", "efficientNetBilberry_weights")
            model.load_state_dict(torch.load(efficientNetBilberry_weights))
        return model


if __name__ == "__main__":
    model = resNetBilberry()

    BATCH_SIZE = 64
    img_test = torch.rand(BATCH_SIZE, 3, 140, 140)
    summary(model, input_size = img_test.shape)
