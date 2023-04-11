import glob

import numpy as np
import scipy as sp

from icecream import ic
import matplotlib.pyplot as plt
import PIL
import torchvision
import torch

import modelBuilder



class Cam_model(torch.nn.Module):
    def __init__(self, model_classifier):
        super(Cam_model, self).__init__()
        self.model_out1 = torch.nn.Sequential(*list(model_classifier.resnet.children())[:-2])
        self.model_avPool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.model_out2 = torch.nn.Sequential(*list(model_classifier.resnet.children())[-1:])
        
    def forward(self, input):
        x_1 = self.model_out1(input)
        x_1b = self.model_avPool(x_1)
        x_2 = self.model_out2(torch.nn.Flatten()(x_1b))
        return x_1, x_2

def plot_CAM(classifier_model, image_value_list, features_list, results_list):
    '''
    Displays the class activation map of a list of images

    Args:
        image_value (list of torch.Tensor): preprocessed input image with size 224 x 224
        features (list of array): features of the image, list of shape (1, 7, 7, 512)
        results (list array): list of output of the sigmoid layer
    '''
    nb_img = len(image_value_list)
    fig = plt.figure(figsize=(25, 15))

    for i, (image_value, features, results) in enumerate(zip(image_value_list, features_list, results_list)):
        fig.add_subplot(1, nb_img, i+1)
        
        # there is only one image in the batch so we index at `0`
        features_for_img = features[0].permute(1,2,0)  # (512, 7, 7) -> (7, 7, 512)
        prediction = results[0]

        # there is only one unit in the output so we get the weights connected to it
        # class_activation_weights = classifier_model.seq[-1].weight[0]
        class_activation_weights = classifier_model.resnet.fc[-1].weight[0]
        class_activation_weights = class_activation_weights.to('cpu').detach().numpy()

        # upsample to the image size
        class_activation_features = sp.ndimage.zoom(features_for_img, (224/7, 224/7, 1/2), order=2)

        # compute the intensity of each feature in the CAM
        cam_output  = np.dot(class_activation_features, class_activation_weights)

        # visualize the results
        plt.axis("off")
        plt.imshow(cam_output, cmap='jet', alpha=0.5)
        plt.imshow(torch.squeeze(image_value).permute(1,2,0), alpha=0.5)


def convert_and_classify(classifier_model, cam_model, img_paths):
    """
    Convert image to tensor and feed to the CAM model.
    Extracting features and results.

    Args:
        classifier_model (torch.nn.Module): trained model
        cam_model (torch.nn.Module): classify activation map model
        image (list of PIL.Image): input images
    """
    features_list = []
    results_list = []
    tensor_img_list = []
    device = next(model.parameters()).device

    # preprocess the image before feeding it to the model
    prep = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    for image in img_paths:
        # load the image
        img = PIL.Image.open(image).convert("RGB")
        tensor_image = torch.as_tensor(prep(img), dtype=torch.float32).unsqueeze(0).to(device)
        tensor_img_list.append(tensor_image.to("cpu"))

        # get the features and prediction
        cam_model.eval()
        with torch.no_grad():
            features, results = cam_model(tensor_image)  # (1, 512, 7, 7)
            features, results = features.to('cpu'), results.to('cpu')
            features_list.append(features)
            results_list.append(results)

    plot_CAM(classifier_model, tensor_img_list, features_list, results_list)


if __name__ == "__main__":
    imgs = [
    'dataset/test_images/fields/4.jpeg',
    # 'dataset/roads/46.jpeg',
    # 'dataset/roads/pexels-photo-775199.jpeg',
    # 'dataset/roads/pexels-photo-209652.jpeg',
    # 'dataset/roads/3b.jpg',
    # 'dataset/roads/6.jpg',
    # 'dataset/roads/15.jpg',
    # 'dataset/roads/iceland-4957449__340.jpg'
    ]

    model = modelBuilder.resNetBilberry(load_resNetBilberry_weights=True)
    cam_model = Cam_model(model)
    convert_and_classify(model, cam_model, imgs)