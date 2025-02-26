# HUMR: Heatmap Uncertainty for Mesh Recovery
This is the official implementation of the WACV 2025 Paper [Utilizing Uncertainty in 2D Pose Detectors for Probabilistic 3D Human Mesh Recovery](https://arxiv.org/abs/2411.16289) by Tom Wehrbein, Marco Rudolph, Bodo Rosenhahn and Bastian Wandt.


## Installation instructions
We recommend creating a clean [conda](https://docs.conda.io/) environment. You can do this as follows:
```
conda env create -f environment.yml
```

A more cross-platform-friendly installation alternative could be to manually follow the commands in `install_env.txt`.


After the installation is complete, you can activate the conda environment by running:
```
conda activate humr
```

## Prepare data
1. [SMPL](https://smpl.is.tue.mpg.de/download.php): Please download <strong>version 1.1.0</strong>, extract it to `data/body_models` and rename the model files to SMPL_NEUTRAL/FEMALE/MALE.pkl.
2. [SMPL-X](https://smpl-x.is.tue.mpg.de/download.php): Please download <strong>version 1.1</strong> and extract it to `data/body_models/smplx`.
3. [SMPL Utils](https://cloud.tnt.uni-hannover.de/index.php/s/WqZs5tLMJqcMj9f): Download <strong>utils.zip</strong> and extract it to `data/utils`.
4. [Pretrained Weights](https://cloud.tnt.uni-hannover.de/index.php/s/WqZs5tLMJqcMj9f): Finally, place all pretrained checkpoints in `data/ckpt`.

Additionally for training and evaluation, please follow the dataset preparation guide (COMING SOON). 

## Demo
We provide a few examples in `data/examples`. Running the following demo will detect each person in each image and then run HUMR for 3D reconstruction. The resulting rendering will be saved to `logs/`
```
python demo.py --cfg configs/default_config.yaml --ckpt data/ckpt/humr_best.ckpt
```

## Training and evaluation
Coming soon...

## Acknowledgements
We benefit from many great resources including but not limited to: [SPIN](https://github.com/nkolot/SPIN), [CLIFF](https://github.com/huawei-noah/noah-research/tree/master/CLIFF), [ViTPose](https://github.com/JunkyByte/easy_ViTPose), [HRNet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch), [BEDLAM](https://github.com/pixelite1201/BEDLAM), [ProHMR](https://github.com/nkolot/ProHMR), [HuManiFlow](https://github.com/akashsengupta1997/HuManiFlow)


## Citing
If you find the model and code useful, please consider citing the following paper:

    @InProceedings{wehrbein25humr,
        author    = {Wehrbein, Tom and Rudolph, Marco and Rosenhahn, Bodo and Wandt, Bastian},
        title     = {Utilizing Uncertainty in 2D Pose Detectors for Probabilistic 3D Human Mesh Recovery},
        booktitle = {Winter Conference on Applications of Computer Vision (WACV)},
        year      = {2025},
    }
