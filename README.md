# Bilberry Assessment 11/04/2023 - Thierry Ksstentini

<p align="center">
  <img src="contents/bilberry_logo.png?raw=true" alt="bilberry" width="200"/>
  <img src="contents/trimble_logo.jpg?raw=true" alt="trimble" width="190"/>
</p>

Welcome to the Bilberry Assessment. This repository brings together my work on field/road classifier and a short overview of an AI paper which has brought (in my opinion) a significant disruption on computer vision.

You'll find **my main work** in this root folder and the **AI paper summary** [here](AIpaper_explained/).


# Table of contents
**[1. Introduction](#intro)**

**[2. Exploration data analysis](#eda)**

**[3. Building the dataset (preprocessing)](#dataset)**

**[4. Building the model](#model)**

  * [4.1. ResNet](#resnet)

  * [4.2. EfficientNet](#efficientnet)

**[5. Training setup](#training)**

**[6. Results](#results)**

**[7. Conclusion](#conclusion)**

**[8. Going Further](#further)**



sumup

# Introduction <a name="intro"></a>
Main Bilberry's problems can be tackled with Computer Vision. The goal here is to handle a supervised machine learning problem to classify **field** and **road** images.

<p align="center">
  <img src="dataset/fields/4.jpg?raw=true" alt="field_ex" width="187"/>
  <img src="dataset/roads/1.jpg?raw=true" alt="road_ex" width="185"/>
</p>

First, I'll expose a short data analysis and the image preprocessing that I choose to face problems that could be encountered like the lack of data which conducts to a model with a weak capacity to generalize on unseen data (overfitting). Then, I'll explain and expose my architecture choices for the building steps of my model and talk about the results I obtained. I'll talk also about the limitations of this project. Finally, I'll expose a way to go further by linking this project with Bilberry's product.

# Exploration data analysis <a name="eda"></a>
The data provided by Bilberry contains a set of field/road folders and a testing folder containing a mix of field an road images. 2 images were mislabeled, I moved them into the right folders. The script of this part is available [here](data_exploration_plots.py). 

### Data balance
At first glance, we see that the data are quite **imbalanced**: road images represent more than 70% of the raw dataset (see piechart). We have to keep that in mind if we want a well trained model than can produce good predictions for both classes.

### Resolution distribution
The **resolution varies a lot** from one image to an other. A barchart shows the resolution distribution as the ratio between the width and the height (outsiders has been removed).

<p align="center">
  <img src="contents/imbalanced_data_piechart.png?raw=true" alt="piechart" width="400"/>
  <img src="contents/WoverH_distribution_imgs.png?raw=true" alt="barchart" width="600"/>
</p>
<p align="center">
    <em> Figure 1: Data and resolution distribution</em>
</p>

### Challenges
One can also talk about some of image that have an ambiguous meaning and may be difficult to learn and classify well. For instance, some field has a well define roadlike path and some road image contans a lot of vegetation.



# Building the dataset (preprocessing) <a name="dataset"></a>
With the data exploratory made previously and the classic machine learning knowledge, this dataset class should handle :

* Data leaking: assure that training and validation images are all differents
* Data imbalanced: making sure to draw as much road image as field image in each batch
* Data resizing: choosing a resolution for output images that will fit with the model
* Data augmentation: vary the data distribution by randomly shift image properties

The first bullet is managed in the `__init__` constructor of the `BilberryDataset` class. The dataset provided is splitted into train/val set which respectively contains 70% and 30% of the road and field images.

The second point comes with a trick in the `__getitem__` method: I use a conditional loop that decide which class image to call based on the index number that will be set by the dataloader when constructing a batch of images. This loop is adjust by the instance variable `ratio=1` i.e. a 1:1 ratio between field and road images.

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


Thirdly, a simple hypothesis I made is to resize images by keeping a W/H ratio equal to the mean of the W/H ratio of the distribution to prevent hard form distortions on the majority of the images. The images are then rescaled with a W/H ratio of 1.5 in the `_preprocess` method, i.e. `H, W = (250, 375)`.

Last but not least, I had to decide a data augmentation strategy to tackle the lack of data. I made multiple observations: 
* horizontal flipping keeps the meaning of an image, 
* we can enlarging the distribution by simulate variation of light, image quality and noisiness (like dust) with the `brightness`, `contrast` and `saturation`. Values have been chosen empiricaly such as it does not break the image meaning, even with the lowest and highest values.


<p align="center">
  <img src="contents/ColorJitter_field.png?raw=true" alt="jitter_field" width="400"/>
  <img src="contents/ColorJitter_road.png?raw=true" alt="jitter_road" width="400"/>
</p>
<p align="center">
    <em> Figure 2: Min/Max values of brightness, constrast and saturation on field and road images</em>
</p>

* a `(224,224)` random crop is applied to the rescaled images. We need a square tensor to feed our model and this randomness permits to force the model to focus and learn specific region of the image. 

<p align="center">
  <img src="contents/RandomCrop_field.png?raw=true" alt="crop_field" width="400"/>
</p>
<p align="center">
    <em> Figure 3: Random crops on the same image</em>
</p>


# Building the model <a name="model"></a>
This part is refering to the modelBuilder [script](modelBuilder.py). Since we do not have a lot of images, I used pretrained model on [ImageNet](https://arxiv.org/pdf/1409.0575.pdf).

## ResNet <a name="resnet"></a>
I decided to take ResNet34 from [*Deep Residual Learning for Image Recognition*](https://arxiv.org/pdf/1512.03385.pdf) as a baseline. I previously worked with this model and it is a well documented one and a lot of projects are based on this one. 

The model architecture is made of the ResNet34 backbone with a head composed of a dropout (for regularisation), followed by two dense layers with a normalization and `ReLU` activation between them. The model outputs a single logit per image turned into probability by a `Sigmoid` activation function. After experiments, I found usefull to unfreeze some backbone layers in addition of the head part. This leads to a model freezed at 80%.


```
=====================================================================================
Layer (type:depth-idx)                        Output Shape              Param #
=====================================================================================
ResNetBilberry                                [1, 1]                    --
├─ResNet: 1-1                                 [1, 1]                    --
│    └─Conv2d: 2-1                            [1, 64, 70, 70]           (9,408)
│    └─BatchNorm2d: 2-2                       [1, 64, 70, 70]           (128)
│    └─ReLU: 2-3                              [1, 64, 70, 70]           --
│    └─MaxPool2d: 2-4                         [1, 64, 35, 35]           --
│    └─Sequential: 2-5 ... 8                  [1, 64, 35, 35]           --
│    │    └─BasicBlock: ...                   [1, ..., ..., ...]        (...,...)
│    └─AdaptiveAvgPool2d: 2-9                 [1, 512, 1, 1]            --
│    └─Sequential: 2-10                       [1, 1]                    --
│    │    └─Dropout: 3-17                     [1, 512]                  --
│    │    └─Linear: 3-18                      [1, 256]                  131,328
│    │    └─LayerNorm: 3-19                   [1, 256]                  512
│    │    └─ReLU: 3-20                        [1, 256]                  --
│    │    └─Linear: 3-21                      [1, 1]                    257
=====================================================================================
Total params: 21,416,769
Trainable params: 4,852,737
Non-trainable params: 16,564,032
=====================================================================================
```


## EfficientNet <a name="efficientnet"></a>
Facing to ResNet34 I choose a most up to date state of the art model. EfficientNet from [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/pdf/1905.11946.pdf) brought efficiency and accuracy upgrades by highlighting the relationship between resolution (input image size), width (number of channels) and depth (number of layers). This, combined with a highly optimized model architecture, takes EfficientNets to be the SOTA way to go (in 2020). Here we'll use the *B0* variation **EfficientNetB0** which should give better accuracy while having fewer parameters to train.

To keep consistency regarding ResNet, roughly 20% of EfficientNetB0 is unfreezed and ready to be finetuned.

```
====================================================================================================
Layer (type:depth-idx)                                       Output Shape              Param #
====================================================================================================
EfficientNetBilberry                                         [1, 1]                    --
├─EfficientNet: 1-1                                          [1, 1]                    --
│    └─Sequential: 2-1                                       [1, 1280, 5, 5]           --
│    │    └─Conv2dNormActivation: 3-1                        [1, 32, 70, 70]           (928)
│    │    └─Sequential: 3-2                                  [1, 16, 70, 70]           (1,448)
│    │    └─Sequential: 3-...                                [1, ..., ..., ...]        (...)
│    │    └─Conv2dNormActivation: 3-9                        [1, 1280, 5, 5]           412,160
│    └─AdaptiveAvgPool2d: 2-2                                [1, 1280, 1, 1]           --
│    └─Sequential: 2-3                                       [1, 1]                    --
│    │    └─Dropout: 3-10                                    [1, 1280]                 --
│    │    └─Linear: 3-11                                     [1, 256]                  327,936
│    │    └─LayerNorm: 3-12                                  [1, 256]                  512
│    │    └─ReLU: 3-13                                       [1, 256]                  --
│    │    └─Linear: 3-14                                     [1, 1]                    257
====================================================================================================
Total params: 4,336,253
Trainable params: 1,110,145
Non-trainable params: 3,226,108
====================================================================================================
```


# Training setup <a name="training"></a>
This part is refering to the [train](train.py) script.
* **Optimizer**: [Adam](https://arxiv.org/pdf/1412.6980.pdf) with a learning rate `lr=0.001` and `weight_decay=0.0005`. It is most of the time a good idea to start with this basic and well known optimizer and those values as a baseline
* **Criterion**: Binary CrossEntropy Loss function `torch.nn.BCELoss`
* **Device**: `mps` or `cuda` if available, else `cpu` (note on mps device [here](https://pytorch.org/docs/stable/notes/mps.html))


# Results <a name="results"></a>

<p align="center">
    <em> Class Activation Map of well predicted field and road images</em>
</p>
<p align="center">
  <img src="contents/CAM_fields.png?raw=true" alt="jitter_field" width="1000"/>
  <img src="contents/CAM_roads.png?raw=true" alt="jitter_road" width="1000"/>
</p>

# Conclusion <a name="conclusion"></a>

# Going further <a name="further"></a>
## Distilled Network
## Pruning
## Float16 learning