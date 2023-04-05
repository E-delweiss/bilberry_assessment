import torch
from torchinfo import summary
from icecream import ic

class ResNetBilberry(torch.nn.Module):
    def __init__(self, pretrained):
        super(ResNetBilberry, self).__init__()
        ### Load ResNet model
        resnet = torchvision.models.resnet50(pretrained=pretrained)
        
        ### Freeze ResNet weights
        if pretrained:
            for param in resnet.parameters():
                param.requires_grad = False

        ### Backbone part
        self.backbone = torch.nn.Sequential(*list(resnet.children())[:-4]) ### ???

        ### Head part
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear() ### ???
        )

    def forward(self, input:torch.Tensor)->tuple:
        """
        TODO
        """     
        x = self.backbone(input)
        x = self.fc(x)
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
    if load_yoloweights:
        resNetBilberry_weights = config.get("WEIGHTS", "resNetBilberry_weights")
        model.load_state_dict(torch.load(resNetBilberry_weights))
    return model


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
    model = NetMNIST(sizeHW=140, S=6, C=10, B=2)

    BATCH_SIZE = 64
    img_test = torch.rand(BATCH_SIZE, 1, 140, 140)
    summary(model, input_size = img_test.shape)
