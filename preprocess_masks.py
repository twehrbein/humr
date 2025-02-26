import glob
from pathlib import Path
import os
import shutil


def preprocess_agora_masks():
    print("preprocess AGORA segmentation masks...")
    base_path = ""  # "/Downloads/tmp/train_masks_1280x720/train/"
    target_path = "data/env_masks/images/"
    Path(target_path).mkdir(parents=True, exist_ok=True)
    for filename in glob.iglob(base_path + '**', recursive=True):
        if os.path.isfile(filename):
            assert filename.endswith("_1280x720.png")
            name = filename.split("_1280x720.png")[0]
            # only use full masks and not single person masks
            if int(name.split("_")[-1]) != 0:
                continue
            outname = Path(filename).name
            outname = outname.replace("_mask_", "_")
            outname = outname.replace("_00", "_0", 1)
            outname = outname.replace("_00000_1280x720.png", "_1280x720_all_person.png")
            # copy to target path
            outpath = os.path.join(target_path, outname)
            shutil.copyfile(filename, outpath)


def preprocess_bedlam_masks():
    print("preprocess BEDLAM segmentation masks...")
    base_path = ""  # "/Downloads/tmp/bedlam_masks/"
    target_path = "data/env_masks/"
    Path(target_path).mkdir(parents=True, exist_ok=True)
    for filename in glob.iglob(base_path + '**', recursive=True):
        if os.path.isfile(filename):
            # only use environment mask
            if not filename.endswith("_env.png"):
                continue
            # only use 6fps version
            idx = int(filename.split("_")[-2])
            if idx % 5 != 0:
                continue
            outname = filename.split(base_path)[-1]
            outname = outname.replace("/masks/", "_6fps/")
            outpath = os.path.join(target_path, outname)
            dirpath = os.path.dirname(outpath)
            Path(dirpath).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(filename, outpath)


if __name__ == "__main__":
    preprocess_agora_masks()
    preprocess_bedlam_masks()
