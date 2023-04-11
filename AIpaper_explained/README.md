# Explained - *An image is worth 16x16 words: transformers for image recognition at scale*

## Motivations
I wanted to talk about this paper because I found it was a new and exciting way to approach computer vision. 

First of all, it talks about pretraining on more than 300M high resolution images ! The fact that we can deal with such an amont of high resolution data is impressive (you would need the computational power of Google, but still). 

Futhermore, this paper tickled my engineer brain: in product engineering, starting a new project comes with high expenses, especially in aeronautics where I come frome. To innovate on a new architecture, one should reconsider the whole process, from research offices to assembly lines. So, the first question is *what do we have "on shelves" that works, and that could fit with this new project ?* That paper uses that line of thinking, it's why I found it so clever. We know that self attention model is a game changer in NLP problems, so the authors came up with this idea *"could we reuse what is actually working well to deal with an other scope of problems ?"*.

## Results
### Interests
* Using the encoder part of a self-attention architecture to classify images
* After pretraining on large amont of data, ViT reaches state of the art accuracy compared to its CNN conterparts on most of image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.)
* This, while requiring fewer computational ressources to train (up to 4 times less)
* It appears to be robust (compared to the SOTA) to various image classification tasks like *Natural*, *Specialized* and *Structured* (see VTAB classification)

### Drawbacks
* It requires a huge amont of data for the pretraining, and lighter versions do not, or struggle to, attain SOTA accuracies
* Usually, architectures are tested on images recognition tasks but also on detection and segmentation. Those last challenges are not explored in this paper
* Even if pretraining would require such expensive computational costs that only few people may want to reproduce it, one can note that the dataset (JFT-300M) used is not open-source and then this paper is not reproducible outside of Google.

## References
Related references you may want to check to have a better understanding of this paper.

* The famous [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762.pdf) paper which establish the use of multi-head self-attention mechanism in NLP
* [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*]() which was the SOTA model for NLP in 2020
* Both paper [*Big Transfer (BiT):
General Visual Representation Learning*](https://arxiv.org/pdf/1912.11370.pdf) and [*Fixing The Train-Test Resolution Discrepancy: FixeEfficientNet*](https://arxiv.org/pdf/2003.08237.pdf) which are the SOTA models in 2020. Those are the main competitors facing ViT.
* The dataset used: [ImageNet-1k](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf), [ImageNet-21k](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [VTAB](https://arxiv.org/pdf/1910.04867.pdf) and the Google in-house [JFT-300](https://arxiv.org/pdf/2106.04560.pdf)
* Finally, I wanted to mention this paper: [Training data-efficient image transformers & distillation through attention](https://arxiv.org/pdf/2012.12877.pdf) which is a direct consequence of ViT and use distilled knowledge to reach ViT performances whilst training on ImageNet only

## Sum up

### Architecture
As mentioned above, ViT models use multi-head self-attention architecture, except it does not need the decoder part since we are only intrested in classifing the output embeddings (see [Figure 1](#fig1)).

<p align="center">
    <img src="contents/NLP2CV_transformer.png?raw=true" width="450" name="fig1"/>
</p>
<p align="center">
    <em> Figure 1: Using Multi-Head self-attention encoder in Computer Vision</em>
</p>

The main difference lies in turning the image into a form compatible with the MHA block. The authors suggest splitting images into fixed-size patches, linearly embedding each of them and add (or concat, it depends of the method) position embeddings (this process is very similar to the one used in NLP problems). Like in NLP, position is important and positional encoding adds learnable parameters to keep in track the spatial position of each patch (see [Figure 2](#fig2)). Note that in a non intuitive manner, positinal encoding is not compulsory, but conducts to higher performances.

<p align="center">
    <img src="contents/ViT.png?raw=true" width="450" name="fig2"/>
</p>
<p align="center">
    <em> Figure 2: Vision Transformer architecture (from paper)</em>
</p>

### Note about ViT
By using ViT, one looses the inductive bias provided by CNNs. The authors explain: "*In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model. In ViT, only MLP layers are local and translationally equivariant, while the self-attention layers are global*". The only 2D information carried by the model, lies in the beggining of the architecture (patch splitting and positional embeddings) and "*all spatial relations between the patches have to be learned from scratch*". Knowing that it appears understandable that ViTs need a large amont of images compared to CNNs.

As an alternative, hybrid architecture could be considered, where input sequences (patches) are a CNN's feature maps. This architecture is also exposed in the paper.

