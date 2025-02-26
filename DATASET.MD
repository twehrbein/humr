# Dataset preparation

## Training data
We use [BEDLAM](https://bedlam.is.tue.mpg.de), [AGORA](https://agora.is.tue.mpg.de) and [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) for training.

To download BEDLAM and AGORA training images, run the following script (adapted from [here](https://github.com/pixelite1201/BEDLAM/blob/master/fetch_training_data.sh)). Since, the data size is huge it will take a while to finish. Please first register on BEDLAM website.
```
bash fetch_training_data.sh
```
Once downloaded, you can uncompress the data in the same directory.

You should then have the following structure in your data folder:
```  
${data}  
|-- training_images
|   |-- 20221010_3_1000_batch01hand_6fps
|   |   |-- png
|   |   |   |-- seq_000000
|   |   |   |-- seq_000001
|   |   |   |-- ...
|   |-- 20221010_3-10_500_batch01hand_zoom_suburb_d_6fps
|   |-- ...
|   |-- images
|   |   |-- ag_trainset_3dpeople_bfh_archviz_5_10_cam00_00000_1280x720.png
|   |   |-- ...
```  

We provide parsed annotation files [here](https://cloud.tnt.uni-hannover.de/index.php/apps/files/files/9595601?dir=/WACV%202025/data/annotations). Please put them in `data/annotations`.
Additionally, to speed up training, we precompute ViTPose predictions and save samples from the predicted heatmaps. They can be downloaded from [here](https://cloud.tnt.uni-hannover.de/index.php/apps/files/files/9595222?dir=/WACV%202025/data) and should be placed in `data/heatmap_samples`.

We use segmentation masks provided by BEDLAM and AGORA during training. Follow these steps to prepare the data:
1. Download <strong>Training Masks 1280x720</strong> from [AGORA](https://agora.is.tue.mpg.de/download.php) and extract it.
2. Follow information provided [here](https://bedlam.is.tue.mpg.de/download.php) to download the BEDLAM segmentation masks and move all downloaded top-level folders to a new directory.
3. Edit `base_path` variables in `preprocess_masks.py` and run `python preprocess_masks.py` to copy the required segmentation masks to the correct location.
4. You can now delete the downloaded masks to free disk space.

```  
${data}  
|-- env_masks
|   |-- 20221010_3_1000_batch01hand_6fps
|   |   |-- seq_000000
|   |   |-- seq_000001
|   |   |-- ...
|   |-- 20221010_3-10_500_batch01hand_zoom_suburb_d_6fps
|   |-- ...
|   |-- images
|   |   |-- ag_trainset_3dpeople_bfh_archviz_5_10_cam00_00000_1280x720_all_person.png
|   |   |-- ...
```  
See below for instructions for 3DPW.

## Evaluation data
We use [3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW/) and [EMDB](https://eth-ait.github.io/emdb/) for evaluation. The image files can be downloaded from their websites.
To evaluate the plausibility of generated hypotheses, we precompute person segmentation masks for a subset of EMDB and provided them [here](https://cloud.tnt.uni-hannover.de/index.php/apps/files/files/9595222?dir=/WACV%202025/data). 

Please place the files such that the data folder has the following structure:
```  
${data}  
|-- test_images
|   |-- 3DPW
|   |   |-- imageFiles
|   |-- EMDB
|   |   |-- masks
|   |   |-- P0
|   |   |-- P1
|   |   |-- ...
```  
