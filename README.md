# Signal Strength Drives Feature Preference in CNN Image Classifiers

This repository contains code to run experiments in the paper "Feature Signal Drives Feature Preference in CNN Image Classifiers." There are three subdirectories in this repository, the contents of which are described below. This code was tested using PyTorch 1.7.

## Synthetic Pairs Matrix

This part of the repository is for running the synthetic pairs matrix experiments in the paper. Here are the commands to run all of the experiments in the paper: 

Pairs Matrix 1
```
python main.py --exp_name pairs_matrix1 --pattern_dir pairs_matrix1 --imgnet_augment
```

Pairs Matrix 2
```
python main.py --exp_name pairs_matrix2 --pattern_dir pairs_matrix2 --imgnet_augment
```

Color Deviation
```
python main.py --exp_name color_deviation_(your epsilon here) --pattern_dir pairs_matrix1 --hue_perturb blue_circle --hue_perturb_val (your epsilon here) --imgnet_augment
```

Color Overlap (pattern dirs are already predefined for these. Some overlap values are included, but if you would like to use different ones, you must create them yourself.)
```
python main.py --exp_name color_overlap_(your overlap here) --pattern_dir color_overlap_(your overlap here) --imgnet_augment
```

Predictivity
```
python3 main.py --exp_name predictivity_(your predictivity here) --pattern_dir pairs_matrix1 --pred_drop blue --pred_drop_val (your predictivity here)
```
When you run one of these experiments, datasets will be created and models trained. Datasets will get created and stored in the directory `./data/exp_name`, trained models will get stored in `./models/exp_name`, and results will appear in `./results/exp_name`. When the experiment is done, there should be a file called `master.csv` in the directory `./results/exp_name` which will contain information including each feature's average preference over the course of the experiment, pixel count, and name. A complete list of commands to generate all data in the paper can be found in the `commands.sh` file in the pairs_matrix_experiments subdirectory.

## Texture Bias

Stimuli and helper code is used from the open-sourced code of the paper "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness" (https://github.com/rgeirhos/texture-vs-shape).

To run the experiments from our paper with an ImageNet-trained ResNet-50, you can do the following:

Normal Texture Bias
```
python main.py
```

Varying degrees of background interpolation to white (use 0 for completely white, 1 for texture background).
```
python main.py --bg_interp (your interpolation here)
```

Resizing
```
python main.py --bg_interp 0 --size (your fraction of the object size here)
```

Landscapes
```
python main.py --bg_interp 0 --landscape
```

Only full shapes
```
python main.py --only_complete
```

Only full shapes masked with masked/interpolated background
```
python main.py --only_complete --bg_interp (your interpolation here)
```

## Excessive Invariance

Running these experiments is a bit more involved. A complete list of commands you must run to reproduce all data and graphs found in the paper can be found in the `commands.sh` file in the excessive_invariance subdirectory. Comments in the file describe what each step represents.