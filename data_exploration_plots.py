import glob
import pandas as pd 
import numpy as np

import PIL
from icecream import ic
import matplotlib.pyplot as plt
import seaborn as sns

################### Pie chart
### Load paths
fields = glob.glob('dataset/fields/*')
roads = glob.glob('dataset/roads/*')

### Define data
data = [len(fields), len(roads)]
labels = ['Fields', 'Roads']

### Define Seaborn color palette to use
colors = sns.color_palette('pastel')[0:5]

### Plot piechart
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.title("Field/Road image distribution")
plt.savefig("contents/imbalanced_data_piechart.png", format="png", transparent=True)
#####################################################


################### H&W distribution
### Load heights and widths
H_size_field = np.array([PIL.Image.open(img).convert("RGB").size[1] for img in fields])
W_size_field = np.array([PIL.Image.open(img).convert("RGB").size[0] for img in fields])
H_size_road = np.array([PIL.Image.open(img).convert("RGB").size[1] for img in roads])
W_size_road = np.array([PIL.Image.open(img).convert("RGB").size[0] for img in roads])

road_img_ratios = W_size_road/H_size_road
field_img_ratios = W_size_field/H_size_field

### Remove outsiders
road_img_ratios = road_img_ratios[(road_img_ratios <= 2)&(road_img_ratios >= 0.7)]
field_img_ratios = field_img_ratios[(field_img_ratios <= 2)&(field_img_ratios >= 0.7)]

### Set fig suplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,3))

### Road images
ax1.bar(range(len(road_img_ratios)), road_img_ratios)
ax1.plot(range(len(road_img_ratios)), np.repeat(np.mean(road_img_ratios), len(road_img_ratios)), color='red')
ax1.set_ylim([0, 2.5])
ax1.set_xlabel("Images")
ax1.set_ylabel("Width over Hight")
ax1.set_xticks([])
ax1.legend([f"mean = {np.mean(road_img_ratios):.2f}"])
ax1.set_title("Distribution of the W/H ratio for 'road' images")

### Field images
ax2.bar(range(len(field_img_ratios)), field_img_ratios)
ax2.plot(range(len(field_img_ratios)), np.repeat(np.mean(field_img_ratios), len(field_img_ratios)), color='red')
ax2.set_ylim([0, 2.5])
ax2.set_xlabel("Images")
ax2.set_ylabel("Width over Hight")
ax2.set_xticks([])
ax2.legend([f"mean = {np.mean(field_img_ratios):.2f}"])
ax2.set_title("Distribution of the W/H ratio for 'field' images")

### Save fig
fig.savefig('contents/WoverH_distribution_imgs.png')
##############