#!/bin/bash

# pairs matrix 1 and 2
python3 main.py --exp_name pairs_matrix1 --pattern_dir pairs_matrix1 --imgnet_augment
python3 main.py --exp_name pairs_matrix2 --pattern_dir pairs_matrix2 --imgnet_augment

# color overlap experiments
python3 main.py --exp_name color_overlap_0 --pattern_dir color_overlap_0 --imgnet_augment
python3 main.py --exp_name color_overlap_25 --pattern_dir color_overlap_25 --imgnet_augment
python3 main.py --exp_name color_overlap_50 --pattern_dir color_overlap_50 --imgnet_augment
python3 main.py --exp_name color_overlap_75 --pattern_dir color_overlap_75 --imgnet_augment
python3 main.py --exp_name color_overlap_90 --pattern_dir color_overlap_90 --imgnet_augment
python3 main.py --exp_name color_overlap_95 --pattern_dir color_overlap_95 --imgnet_augment

# predictivity experiments
python3 main.py --exp_name predictivity_50 --pattern_dir pairs_matrix1 --imgnet_augment --pred_drop blue --pred_drop_val 0.5
python3 main.py --exp_name predictivity_75 --pattern_dir pairs_matrix1 --imgnet_augment --pred_drop blue --pred_drop_val 0.75
python3 main.py --exp_name predictivity_90 --pattern_dir pairs_matrix1 --imgnet_augment --pred_drop blue --pred_drop_val 0.9
python3 main.py --exp_name predictivity_100 --pattern_dir pairs_matrix1 --imgnet_augment --pred_drop blue --pred_drop_val 1.0

# color deviation experiments
python3 main.py --exp_name color_deviation_10 --pattern_dir pairs_matrix1 --imgnet_augment --color_dev blue --color_dev_eps 0.1
python3 main.py --exp_name color_deviation_25 --pattern_dir pairs_matrix1 --imgnet_augment --color_dev blue --color_dev_eps 0.25
python3 main.py --exp_name color_deviation_50 --pattern_dir pairs_matrix1 --imgnet_augment --color_dev blue --color_dev_eps 0.5
python3 main.py --exp_name color_deviation_75 --pattern_dir pairs_matrix1 --imgnet_augment --color_dev blue --color_dev_eps 0.75