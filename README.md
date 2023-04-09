# Bilberry Assessment

<p align="center">
  <img src="contents/bilberry_logo.png?raw=true" alt="bilberry" width="200"/>
  <img src="contents/trimble_logo.jpg?raw=true" alt="trimble" width="190"/>
</p>

Welcome to the Bilberry Assessment. This repository brings together my work on field/road classifier and a short overview of an AI paper which has brought (I think) a significant disruption on computer vision.

You'll find my main work in the root folder and the AI paper sumup here[ADD PATH]. Finally, I give an overview of one way to go further here[ADD PATH] by thinking about a challenge that the Bilberry's product might overcome. 





# Introduction
Main Bilberry's problems can be tackled with Computer Vision. The goal here is to handle a supervised machine learning problem to classify **field** and **road** images.

<p align="center">
  <img src="dataset/fields/4.jpg?raw=true" alt="field_ex" width="187"/>
  <img src="dataset/roads/1.jpg?raw=true" alt="road_ex" width="185"/>
</p>

First, I'll expose a short data analysis and the image preprocessing that I choose to handle various problems that could be encountered like the lack of data which conduct for sure to a model with a weak capacity to generalize on unseen data (also known as overfitting). Then, I'll explain and expose my architecture choices for the building steps of my model and talk about the results I obtained compare with some online work I found. I'll talk also about the limitations of this project which is seriously lacking of data. Finally, I'll show an other PoC than can be related to Bilberry's product challenge, namely, computational cost.

# Exploration data analysis
The data provided by Bilberry contains a set of field/road folders and a testing folder containing a mix of field an road images. 

### Data balance
At first glance, we see that the data are quite **unbalanced**: road images represent more than 70% of the raw dataset (see piechart). We have to keep that in mind if we want a well trained model than can produce good predictions for both classes.

### Resolution distribution
The **resolution varies a lot** from one image to an other. A barchart shows the resolution distribution as the ratio between the width and the height (outsiders has been removed). A simple hypothesis I made is to resize image by the mean of the W/H ratios to prevent hard form deformations on the maximum of images. To simplify the process, I decided to resize images with a **W/H ratio of 1.5** before 



<p align="center">
  <img src="contents/imbalanced_data_piechart.png?raw=true" alt="piechart" width="400"/>
  <img src="contents/WoverH_distribution_imgs.png?raw=true" alt="piechart" width="600"/>
</p>