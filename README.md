# RAST-4.0
RAST4.0:Restorable Arbitrary Style Transfer via Content Leakage Correction

## Requirements  
- python 3.8
- PyTorch 1.8.0
- CUDA 11.1

## Model Testing
- Create ''model'', ''content'' and ''style'' folders under specific training strategy folder.
- Download [VGG pretrained](https://drive.google.com/file/d/1cI6ubAziMdOsSJZEvfofW-iCtnCmsONL/view?usp=share_link) model to ''model'' folder.
- Put testing content images to ''content'' folder.
- Put testing style images to ''style'' folder.
- Run the following command:
```
python eval.py --content_dir ./content/ --style_dir ./style/
```
- The path names for some testing code sections are different. Please modify the path names based on your current specific path.
  
## Model Training
- Create ''model'', ''coco_train'' and ''wiki_train'' folder under specific training strategy folder.
- Download [VGG pretrained](https://drive.google.com/file/d/1cI6ubAziMdOsSJZEvfofW-iCtnCmsONL/view?usp=share_link) model to ''model'' folder.
- Download COCO2014 dataset to ''coco_train'' folder
- Download Wiki dataset to ''wiki_train'' folder
- Run the following command:
```
python train.py --content_dir ./coco_train/ --style_dir ./wiki_train/
```
