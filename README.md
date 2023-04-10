# Bilberry Assessment

<p align="center">
  <img src="contents/bilberry_logo.png?raw=true" alt="bilberry" width="200"/>
  <img src="contents/trimble_logo.jpg?raw=true" alt="trimble" width="190"/>
</p>

Welcome to the Bilberry Assessment. This repository brings together my work on field/road classifier and a short overview of an AI paper which has brought (I think) a significant disruption on computer vision.

You'll find **my main work** in this root folder and the **AI paper sumup** here[ADD PATH]. Finally, I give an overview of one way to **go further** here[ADD PATH] by thinking about a challenge that the Bilberry's product might overcome. 





# Introduction
Main Bilberry's problems can be tackled with Computer Vision. The goal here is to handle a supervised machine learning problem to classify **field** and **road** images.

<p align="center">
  <img src="dataset/fields/4.jpg?raw=true" alt="field_ex" width="187"/>
  <img src="dataset/roads/1.jpg?raw=true" alt="road_ex" width="185"/>
</p>

First, I'll expose a short data analysis and the image preprocessing that I choose to handle various problems that could be encountered like the lack of data which conduct for sure to a model with a weak capacity to generalize on unseen data (also known as overfitting). Then, I'll explain and expose my architecture choices for the building steps of my model and talk about the results I obtained compare with some online work I found. I'll talk also about the limitations of this project which is seriously lacking of data. Finally, I'll show an other PoC than can be related to Bilberry's product challenge, namely, computational cost.

# Exploration data analysis
The data provided by Bilberry contains a set of field/road folders and a testing folder containing a mix of field an road images. 2 images were mislabeled, I moved them into the right folders. The script of this part is available [here](data_exploration_plots.py). 

### Data balance
At first glance, we see that the data are quite **imbalanced**: road images represent more than 70% of the raw dataset (see piechart). We have to keep that in mind if we want a well trained model than can produce good predictions for both classes.

### Resolution distribution
The **resolution varies a lot** from one image to an other. A barchart shows the resolution distribution as the ratio between the width and the height (outsiders has been removed). A simple hypothesis I made is to resize images by keeping a W/H ratio equal to the mean of the W/H ratio of the distribution to prevent hard form distortions on the majority of the images.

<p align="center">
  <img src="contents/imbalanced_data_piechart.png?raw=true" alt="piechart" width="400"/>
  <img src="contents/WoverH_distribution_imgs.png?raw=true" alt="barchart" width="600"/>
</p>

### Challenges
One can also talk about some of image that have an ambiguous meaning and may be difficult for a basic model to learn and classify well. For instance, some field has a well define path in the center of the image (could be identify as a road) and some road has some vegetation on a fair part of the image.



# Building our dataset (preprocessing)
I use Pytorch for this project. Hence, I built a [custom dataset](bilberry_dataset.py) class named `BilberryDataset` with the `torch.utils.data.Dataset` module which will be loaded as a `torch.utils.data.DataLoader`. With the data exploratory made previously and the classic machine learning knowledge, this dataset class should handle :

* Data leaking: assure that training and validation images are all differents
* Data imbalanced: making sure to draw as much road image as field image in each batch
* Data resizing: choosing a resolution for output images that will fit with the model
* Data augmentation: vary the data distribution by randomly shift image properties

The first bullet is managed in the `__init__` constructor of the `BilberryDataset` class. The dataset provided is split into train/val set which respectively contains 70% and 30% of the road and field images.

The second point comes with a trick in the `__getitem__` method: I use a conditional loop that decide which class image to call based on the index number that will be set by the dataloader when it constructs a batch of images. This loop is adjust by the instance variable `ratio=1` i.e. a 1:1 ratio between field and road images.

```python
if self.ratio:
    ### Create a new idx regarding the asked ratio
    pos_idx = idx // (self.ratio + 1)

    ### Select img/label regarding the ratio
    if idx % (self.ratio + 1):
        neg_idx = idx - 1 - pos_idx
        neg_idx = neg_idx % len(self.imgset_roads)
        img_path = self.imgset_roads[neg_idx]
        label = self.label_roads[neg_idx]
    else:
        pos_idx = pos_idx % len(self.imgset_fields)
        img_path = self.imgset_fields[pos_idx]
        label = self.label_fields[pos_idx]
```

Thirdly, I decided to rescale images with a W/H ratio of 1.5 to distort images to a minimum. The images are then rescale to `H, W = (250, 375)` in the `_preprocess` method.

Last but not least, I had to decide to a data augmentation strategy to tackle the lack of data. I made multiple observations: 
* horizontal flipping keeps the meaning of an image whereas it is a field or road image, 
* to enlarge the distribution and predict variation of light, image quality and noisiness (like dust) we can play with the `brightness`, the `contrast` and the `saturation`. Values have been chose empiricaly such as it does not break the image meaning, even with the lower and higher values.

<p align="center">
  <img src="contents/ColorJitter_field.png?raw=true" alt="jitter_field" width="400"/>
  <img src="contents/ColorJitter_road.png?raw=true" alt="jitter_road" width="400"/>
</p>

* a random crop of a square size `224` is applied to the rescaled images since we need a square tensor to feed our model. The randomness permits to force the model to focus and learn specific region of the image one at a time. 

<p align="center">
  <img src="contents/RandomCrop_field.png?raw=true" alt="crop_field" width="400"/>
</p>


# Building the model
This part is refering to the [modelBuilder](modelBuilder.py) script. Since we do not have a lot of images, I used pretrained model on ImageNet from [ImageNet Large Scale Visual Recognition Challenge](https://arxiv.org/pdf/1409.0575.pdf).
## ResNet
I decided to take ResNet34 from [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385.pdf) as a baseline. I previously worked with this model and it is a well documented one and a lot of projects are based on this one. 

The model architecture is made of the ResNet34 backbone with a head composed of a dropout[ON GARDE ???] (for regularisation), followed by two dense layers with a normalization and `ReLU` activation between them. The model outputs a single logit per image turned into probability by a `Sigmoid` activation function.

## EfficientNet
Facing to ResNet34 I choose a most up to date state of the art model. EfficientNet from [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) brought a lot of efficiency and accuracy upgrades by highlighting the relationship between resolution (input image size), width (number of channels) and depth (number of layers). This, combine with a highly optimize model architecture, leads to variations of EfficientNets. Here we'll use **EfficientNetB4** which gives better accuracy on ImageNet while have a close number of parameters than ResNet34.

<p align="center">
    <img src="contents/efficientnet_paper_comparisons.png?raw=true" width="400"/>
</p>
<p align="center">
    <em> Model Size vs. ImageNet Accuracy from EfficientNet paper </em>
</p>


# Training setup
This part is refering to the [train](train.py) script.
* **Optimizer**: Adam from [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/pdf/1412.6980.pdf) with a learning rate `lr=0.001` and `weight_decay=0.0005`. It is most of the time a good idea to start with this basic and well known optimizer and those values as a baseline.
* **Criterion**: Binary CrossEntropy Loss function `torch.nn.BCELoss``
* **Device**: `mps` or `cuda` if available, else `cpu` (note on mps device [here](https://pytorch.org/docs/stable/notes/mps.html))


# Results