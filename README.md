# LDRloss
<img src="algorithm_overview.png">
LDR loss for image enhancement network<br/>
 └ Layered Difference Representation loss

# Tested Environment 
* Python = 3.12.1
* Torch = 2.2.0 including cuda 12.1 and cudnn 8.0

# Datasets
MIT-Adobe 5K: Original link is <a href="https://data.csail.mit.edu/graphics/fivek/"> here </a> <br/>
LOL-v2: Original link is <a href="https://github.com/flyywh/CVPR-2020-Semi-Low-Light"> here </a> <br/>
Mixed dataset is comprised of LOL-v2 and FiveK (we update this soon)

# Train
you can train using LDRM_NET.py
train dataset should be include "input and gt folder" <br/>
e.g.) train/input<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;               /gt 

# Test 
You can test using Test.py

# Pretrained Model
Two pretrained model files are uploaded to models_pretrained folder. 
 1. train env.: 1000epoch, alpha 20
 2. train env.: 500epoch, alpha 10<br/>
 
You may use pretrained model file by copying "final_model.pth" to models folder.

# Citation 
We will update soon. 
