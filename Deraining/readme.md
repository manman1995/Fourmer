## Applications
###  Image deraining
####  Prepare data
Download the training data and add the data path to the config file (/basicsr/option/train/RAIN200H(L)/*.yml). 
#### : Training
```
python /basicsr/train.py -opt options/train/RAIN200H/fourmer.yml
python /basicsr/train.py -opt /LLIE/options/train/RAIN200L/fourmer.yml
```

