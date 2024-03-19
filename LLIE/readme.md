## Applications
###  Low-Light Image Enhancement
####  Prepare data
Download the training data and add the data path to the config file (/basicsr/option/train/LLIE/*.yml). Please refer to [LOL](https://daooshee.github.io/BMVC2018website/) and [Huawei](https://drive.google.com/drive/folders/1rFUSdcw833haZfkGKODvu1hrv2pgxYT_?usp=drive_link) (it includes 2480 images, and we we randomly select 2200 images for training and the remaining 280 for testing) for data download. 
#### Training
```
python /LLIE/train.py -opt options/train/train_Enhance.yml
```
