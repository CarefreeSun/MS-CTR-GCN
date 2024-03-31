# Multi-Stream CTR-GCN
This repo is the implementation of our method for [2024 ICME Grand Chanllenge Multi-Modal Video Reasoning and Analyzing Competition](https://sutdcv.github.io/MMVRAC/), Track #10: Skeleton-based Action Recognition(UAV-Human dataset). 

The results are listed below:

| Split | Acc(%) |
| :---: | :----: |
|  v1   | 49.05  |
|  v2   | 75.61  |

## Train the model
We provide scripts for you to train the three basic streams conveniently.
```bash
$ sh train_uav_joint.sh
$ sh train_uav_bone.sh
$ sh train_uav_motion.sh
```
To utilize the long-tail loss, please refer to the code in [main_logit_adjust.py](main_logit_adjust.py).

## Multi-stream ensemble
We provide the best scores and their corresponding weights in the [./weights](./weights/) directory. You can get the accuracies with the commands below:
```bash
$ sh ensemble_v1.sh
$ sh ensemble_v2.sh
```
Moreover, your own weights and scores would be stored in the ./workdir directory. You could test your results by modifying the [ensemble_v1.sh](ensemble_v1.sh) and [ensemble_v2.sh](ensemble_v2.sh).

## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).

Thanks to the original authors for their work!