# Explained - *An image is worth 16x16 words: transformers for image recognition at scale*

## Motivations
I wanted to talk about this paper because I found it so clever and I was so exciting when I read it. 

First of all, it talks about pretraining on more than 300M high resolution images ! The fact that we can deal with a such amont of high resolution data is impressive to me (ok, we need the computational power of Google but still). 

Then, this paper trigger my engineer nature: in product engineering, starting a new project comes with high expenses especially in aeronautic where I come frome. To innovate on a new architecture one should reconsider the whole process, from research offices to assembly lines. So, the first question is *what do we have "on shelves" that works, and that could fit with this new project ?*. I see this paper exactly like that, it's why I found it so clever. One knew that self attention models was a game changer in NLP problems so the authors came with this idea *"could we reuse what is actually working well to deal with an other scope of problem ?"*.

## Results
### Interests
* Using the encoder part of a self-attention architecture to classify images
* After pretraining on large amont of data, ViT reaches state of the art accuracy compared to its CNN conterparts on most of image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.)
* This, while requiring fewer computational ressources to train (up to 4 times less)
* It appears to be rebust (compared to the SOTA) to various image classification tasks like *Natural*, *Specialized* and *Structured* (see VTAB classification)

### Drawbacks
* It requires a huge amont of data for pretraining and lighter versions do not or struggle to attain SOTA accuracies
* Usually, architectures are tested on images recognition tasks but also on detection and segmentation. Those last challenges are not explored on this paper
* Even if pretraining would require such expensive computational costs that only few people may want to reproduce the pretraining, one can note that the dataset (JFT-300M) used is not open-source and then this paper is not reproducible outside of Google.

## References
Related references you may want to check to have a better understanding of this paper.

* The famous [*Attention Is All You Need*](https://arxiv.org/pdf/1706.03762.pdf) paper which establish the use of multi-head self-attention mechanism in NLP
* [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*]() which was the SOTA model for NLP in 2020
* Both paper [*Big Transfer (BiT):
General Visual Representation Learning*](https://arxiv.org/pdf/1912.11370.pdf) and [*Fixing The Train-Test Resolution Discrepancy: FixeEfficientNet*](https://arxiv.org/pdf/2003.08237.pdf) which are the SOTA models in 2020. Those are the main competitors facing ViT.
* The dataset used: [ImageNet-1k](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf), [ImageNet-21k](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf), [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html), [Oxford-IIIT Pets](https://www.robots.ox.ac.uk/~vgg/data/pets/), [Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), [VTAB](https://arxiv.org/pdf/1910.04867.pdf) and the Google in-house [JFT-300](https://arxiv.org/pdf/2106.04560.pdf)

## Sum up
