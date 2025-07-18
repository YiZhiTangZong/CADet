# Implementation of cadet



## Train and Test 
Requirements
```
Python 3.6
Pytorch 1.9.1
```
### dataset
Please download related datasets:

Data folder structure:  

```
data/
├── split
│    ├──crack500
│    │    ├── train.txt
│    │    └── test.txt
│    └──...
└── other data process files
datasets/
├── crack500
│   ├── images
│   │   ├── img1.tif
│   │   └── ...
│   ├── annotation_mask
│   │   ├── img1.png
│   │   └── ...
│   ├── annotation_boundary
│   │   ├── img1.png
│   │   └── ...
│   └── others
└── ...
```
We provide our script to generate boundary labels from annotations, your can run (Replace the folder path in brackets below):
```
python tools/produce_boundary.py --mask [annotation folder path] --save [generated boundary labels' path]
```
### train
Please modify the config and run:  
`sh train_cadet.sh`
### test
Please modify the config and run:   
`sh test_cadet.sh`  
