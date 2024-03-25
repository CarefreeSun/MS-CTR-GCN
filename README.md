# Multi-Stream CTR-GCN
This repo is the implementation of our method for [2024 ICME Grand Chanllenge Multi-Modal Video Reasoning and Analyzing Competition](https://sutdcv.github.io/MMVRAC/), Track #10: Skeleton-based Action Recognition(UAV-Human dataset). 

## Train the model
We provide scripts for you to train the model conveniently.
```bash
$ sh train_uav_joint.sh
$ sh train_uav_bone.sh
$ sh train_uav_motion.sh
```

## Multi-stream ensemble
We provide the best scores and their corresponding weights in the [./weights](./weights/) directory. You can get the accuracy with the command below:
```bash
$ sh ensemble.sh
```
Moreover, your own weights would be stored in the ./workdir directory. You could test your results by modifying the [ensemble.sh](ensemble.sh) file.

## Acknowledgements

This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN).

Thanks to the original authors for their work!