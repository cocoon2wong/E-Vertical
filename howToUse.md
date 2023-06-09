---
layout: page
add-md-links: true
title: Codes Guidelines for the E-V^2-Net
subtitle: "Official implementation of the paper \"Another Vertical View: A Hierarchical Network for Heterogeneous Trajectory Prediction via Spectrums\""
gh-repo: cocoon2wong/E-Vertical
gh-badge: [star, fork]
---
<!--
 * @Author: Conghao Wong
 * @Date: 2023-02-27 16:20:22
 * @LastEditors: Conghao Wong
 * @LastEditTime: 2023-06-09 11:24:35
 * @Description: file content
 * @Github: https://cocoon2wong.github.io
 * Copyright 2023 Conghao Wong, All Rights Reserved.
-->

## Get Started

---

You can clone [this repository](https://github.com/cocoon2wong/E-Vertical) by the following command:

```bash
git clone https://github.com/cocoon2wong/E-Vertical.git
```

Since the repository contains all the dataset files, this operation may take a longer time.
Or you can just download the zip file from [here](https://codeload.github.com/cocoon2wong/E-Vertical/zip/refs/heads/main).

## Requirements

---

The codes are developed with python 3.9.
Additional packages used are included in the `requirements.txt` file.

{: .box-warning}
**Warning:** We recommend installing all required python packages in a virtual environment (like the `conda` environment).
Otherwise, there *COULD* be other problems due to the package version conflicts.

Run the following command to install the required packages in your python environment:

```bash
pip install -r requirements.txt
```

Read our post for more information about the environment configurations:

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://cocoon2wong.github.io/2022-03-03-env/">💡 Environment Configuration Guidelines</a>
</div>

## Dataset Prepare and Process

---

Before training `E-V^2-Net` on your own dataset, you should add your dataset information.
See [this document](https://cocoon2wong.github.io/Project-Luna/) for details.

## Training

---

Due to the difference in the target trainable variables of loss functions in different subnetworks, we divide the network into two parts to train them separately for the convenience of training.
The network can still be used as an end-to-end network during testing.

### Stage-1 Subnetwork

It is the coarse-level keypoints estimation sub-network.
To train the subnetwork, you can pass the --model va argument to run the `main.py`.
You should also specify the temporal keypoint indexes in the predicted period.
For example, when you want to train a model that predicts future 12 frames of trajectories, and you would like to set $N_{key} = 3$ (which is the same as the basic settings in our paper), you can pass the `--key_points 3_7_11` argument when training.
Please note that indexes start with `0`.
You can also try any other keypoints settings or combinations to train the subnetwork and obtain it that best fits your datasets.
Please refer to section "Args Used" to learn how other args work when training and evaluating.

{: .box-warning}
**Warning:** Do not pass any value to `--load` when training, or it will start evaluating the loaded model.

For a quick start, you can train the subnetwork via the following minimum arguments:

```bash
python main.py --model eva --key_points 3_7_11 --T fft --split sdd
```

### Stage-2 Subnetwork

It is the fine-level spectrum interpolation sub-network.
You can pass the `--model vb` to run the training with the following minimum arguments:

```bash
python main.py --model vb --points 3 --split sdd
```

## Evaluation

---

You can use the following command to evaluate the `E-V^2-Net` performance end-to-end:

```bash
python main.py \
  --model MKII \
  --loada A_MODEL_PATH \
  --loadb B_MODEL_PATH
```

Where `A_MODEL_PATH` and `B_MODEL_PATH` are the folders of the two sub-networks' weights.

### Pre-Trained Models

We have provided our pre-trained model weights to help you quickly evaluate the `E-V^2-Net` performance.
Our pre-trained models contain:

- 2D coordinate prediction model trained on `ETH-UCY` (`leave-one-out` strategy) and `SDD` with the prediction horizon of 3.2-4.8 seconds;
- 2D bounding box prediction model trained on `SDD (2D bounding box)` with the prediction horizon of 3.2-4.8 seconds;
- 3D bounding box prediction model trained on `nuScenes (3D bounding box)` with the prediction horizon of 2.0-2.0 seconds.

Click the following buttons to download our weights and learn about how to install these datasets.
We recommend that you download the weights and place them in the `weights/silverballers` folder.

<div style="text-align: center;">
    <a class="btn btn-colorful btn-lg" href="https://github.com/cocoon2wong/E-Vertical/releases/tag/V1.0">⬇️ Download Weights</a>
    <a class="btn btn-colorful btn-lg" href="https://cocoon2wong.github.io/Project-Luna/howToUse/">💡 Dataset Guidelines</a>
</div>

You can start the quick evaluation via the following commands:

```bash
python main.py --model MKII --loada $MODEL_PATH_A --loadb $MODEL_PATH_B
```

Here, `$MODEL_PATH_A` is the path of the first-stage sub-network (such as `weights/silverballers/EV_co_DFT_sdd`), and `$MODEL_PATH_B` is the corresponding second-stage sub-network's weights path.

For example, you can run the following commands to run evaluations on `ETH-UCY` benchmark to validate our 2D coordinate prediction performance:

```bash
for dataset in eth hotel univ zara1 zara2
    python main.py --model MKII \
        --loada weights/silverballers/EV_co_DFT_${dataset} \
        --loadb weights/silverballers/VB_co_DFT_${dataset}
```

{: .box-warning}
**Warning:** For 3D bounding box or higher dimensional prediction models, we do not provide weights for the corresponding stage 2 subnetwork.
You can get the complete prediction results by using the following linear interpolation second stage network by passing `--loadb l` instead.

### Linear-Interpolation Models

You can also start testing the fast version of our pre-trained models with the argument `--loadb l` instead of the `--loadb $MODEL_PATH_B`.
When passing the `--loadb l` argument, it will replace the original stage-2 spectrum interpolation sub-network with the simple linear interpolation method.
Although it may reduce the prediction performance, the model will implement much faster.
You can start testing these linear-interpolation models with the following command:

```bash
python main.py --model MKII --loada $MODEL_PATH_A --loadb l
```

Here, `$MODEL_PATH_A` is still the path of model weights of the stage-1 sub-networks.

### Visualization

If you have the dataset videos and put them into the `videos` folder, you can draw the visualized results by adding the `--draw_reuslts $SOME_VIDEO_CLIP` argument.

{: .box-warning}
**Warning:** You must put videos according to the `video_path` item in the clip's `plist` config file in the `./dataset_configs` folder if you want to draw visualized results on them.

{: .box-warning}
**Warning:** Currently, only 2D coordinate and 2D bounding box visualization predictions are supported for drawing visualizations.

If you want to draw visualized trajectories like what our paper shows, you can add the additional `--draw_distribution 2` argument.
For example, if you have put the video `zara1.mp4` into `./videos/zara1.mp4`, you can draw the `E-V^2-Net` results with the following command:

```bash
python main.py --model MKII \
    --loada ./weights/silverballers/EV_co_DFT_zara1 \
    --loadb ./weights/silverballers/VB_co_DFT_zara1 \
    --draw_results zara1 \
    --draw_distribution 2
```

## Args Used

---

Please specify your customized args when training or testing your model in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 -SHORT_ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.

About the `argtype`:

- Args with argtype=`static` can not be changed once after training.
  When testing the model, the program will not parse these args to overwrite the saved values.
- Args with argtype=`dynamic` can be changed anytime.
  The program will try to first parse inputs from the terminal and then try to load from the saved JSON file.
- Args with argtype=`temporary` will not be saved into JSON files.
  The program will parse these args from the terminal at each time.

<!-- DO NOT CHANGE THIS LINE -->

### Basic args

- `--K_train`: type=`int`, argtype=`static`.
  The number of multiple generations when training. This arg only works for multiple-generation models. 
  The default value is `10`.
- `--K`: type=`int`, argtype=`dynamic`.
  Number of multiple generations when testing. This arg only works for multiple-generation models. 
  The default value is `20`.
- `--anntype`: type=`str`, argtype=`static`.
  Model's predicted annotation type. Can be `'coordinate'` or `'boundingbox'`. 
  The default value is `coordinate`.
- `--batch_size` (short for `-bs`): type=`int`, argtype=`dynamic`.
  Batch size when implementation. 
  The default value is `5000`.
- `--dataset`: type=`str`, argtype=`static`.
  Name of the video dataset to train or evaluate. For example, `'ETH-UCY'` or `'SDD'`. NOTE: DO NOT set this argument manually. 
  The default value is `Unavailable`.
- `--draw_distribution` (short for `-dd`): type=`int`, argtype=`temporary`.
  Controls if draw distributions of predictions instead of points. If `draw_distribution == 0`, it will draw results as normal coordinates; If `draw_distribution == 1`, it will draw all results in the distribution way, and points from different time steps will be drawn with different colors. 
  The default value is `0`.
- `--draw_exclude_type` (short for `-det`): type=`str`, argtype=`temporary`.
  Draw visualized results of agents except user-assigned types. If the assigned types are `"Biker_Cart"` and the `draw_results` or `draw_videos` is not `"null"`, it will draw results of all types of agents except "Biker" and "Cart". It supports partial match and it is case-sensitive. 
  The default value is `null`.
- `--draw_index`: type=`str`, argtype=`temporary`.
  Indexes of test agents to visualize. Numbers are split with `_`. For example, `'123_456_789'`. 
  The default value is `all`.
- `--draw_results` (short for `-dr`): type=`str`, argtype=`temporary`.
  Controls whether to draw visualized results on video frames. Accept the name of one video clip. The codes will first try to load the video file according to the path saved in the `plist` file (saved in `dataset_configs` folder), and if it loads successfully it will draw the results on that video, otherwise it will draw results on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_results` if `draw_results != 'null'`. 
  The default value is `null`.
- `--draw_videos`: type=`str`, argtype=`temporary`.
  Controls whether draw visualized results on video frames and save as images. Accept the name of one video clip. The codes will first try to load the video according to the path saved in the `plist` file, and if successful it will draw the visualization on the video, otherwise it will draw on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_videos` if `draw_videos != 'null'`. 
  The default value is `null`.
- `--epochs`: type=`int`, argtype=`static`.
  Maximum training epochs. 
  The default value is `500`.
- `--force_anntype`: type=`str`, argtype=`temporary`.
  Assign the prediction type. It is now only used for silverballers models that are trained with annotation type `coordinate` but want to test on datasets with annotation type `boundingbox`. 
  The default value is `null`.
- `--force_clip`: type=`str`, argtype=`temporary`.
  Force test video clip (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--force_dataset`: type=`str`, argtype=`temporary`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--force_split`: type=`str`, argtype=`temporary`.
  Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`. 
  The default value is `null`.
- `--gpu`: type=`str`, argtype=`temporary`.
  Speed up training or test if you have at least one NVidia GPU. If you have no GPUs or want to run the code on your CPU, please set it to `-1`. NOTE: It only supports training or testing on one GPU. 
  The default value is `0`.
- `--interval`: type=`float`, argtype=`static`.
  Time interval of each sampled trajectory point. 
  The default value is `0.4`.
- `--load` (short for `-l`): type=`str`, argtype=`temporary`.
  Folder to load model (to test). If set to `null`, the training manager will start training new models according to other given args. 
  The default value is `null`.
- `--log_dir`: type=`str`, argtype=`static`.
  Folder to save training logs and model weights. Logs will save at `args.save_base_dir/current_model`. DO NOT change this arg manually. (You can still change the path by passing the `save_base_dir` arg.) 
  The default value is `Unavailable`.
- `--lr`: type=`float`, argtype=`static`.
  Learning rate. 
  The default value is `0.001`.
- `--model_name`: type=`str`, argtype=`static`.
  Customized model name. 
  The default value is `model`.
- `--model`: type=`str`, argtype=`static`.
  The model type used to train or test. 
  The default value is `none`.
- `--obs_frames` (short for `-obs`): type=`int`, argtype=`static`.
  Observation frames for prediction. 
  The default value is `8`.
- `--pmove`: type=`int`, argtype=`static`.
  Index of the reference point when moving trajectories. 
  The default value is `-1`.
- `--pred_frames` (short for `-pred`): type=`int`, argtype=`static`.
  Prediction frames. 
  The default value is `12`.
- `--protate`: type=`float`, argtype=`static`.
  Reference degree when rotating trajectories. 
  The default value is `0.0`.
- `--pscale`: type=`str`, argtype=`static`.
  Index of the reference point when scaling trajectories. 
  The default value is `autoref`.
- `--restore_args`: type=`str`, argtype=`temporary`.
  Path to restore the reference args before training. It will not restore any args if `args.restore_args == 'null'`. 
  The default value is `null`.
- `--restore`: type=`str`, argtype=`temporary`.
  Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`. 
  The default value is `null`.
- `--save_base_dir`: type=`str`, argtype=`static`.
  Base folder to save all running logs. 
  The default value is `./logs`.
- `--split` (short for `-s`): type=`str`, argtype=`static`.
  The dataset split that used to train and evaluate. 
  The default value is `zara1`.
- `--start_test_percent`: type=`float`, argtype=`static`.
  Set when (at which epoch) to start validation during training. The range of this arg should be `0 <= x <= 1`. Validation may start at epoch `args.epochs * args.start_test_percent`. 
  The default value is `0.0`.
- `--step`: type=`int`, argtype=`dynamic`.
  Frame interval for sampling training data. 
  The default value is `1`.
- `--test_mode`: type=`str`, argtype=`temporary`.
  Test settings. It can be `'one'`, `'all'`, or `'mix'`. When setting it to `one`, it will test the model on the `args.force_split` only; When setting it to `all`, it will test on each of the test datasets in `args.split`; When setting it to `mix`, it will test on all test datasets in `args.split` together. 
  The default value is `mix`.
- `--test_step`: type=`int`, argtype=`static`.
  Epoch interval to run validation during training. 
  The default value is `1`.
- `--update_saved_args`: type=`int`, argtype=`temporary`.
  Choose whether to update (overwrite) the saved arg files or not. 
  The default value is `0`.
- `--use_extra_maps`: type=`int`, argtype=`dynamic`.
  Controls if uses the calculated trajectory maps or the given trajectory maps. The training manager will load maps from `./dataset_npz/.../agent1_maps/trajMap.png` if set it to `0`, and load from `./dataset_npz/.../agent1_maps/trajMap_load.png` if set this argument to `1`. 
  The default value is `0`.

### Silverballers args

- `--Kc`: type=`int`, argtype=`static`.
  The number of style channels in `Agent` model. 
  The default value is `20`.
- `--T`: type=`str`, argtype=`static`.
  Type of transformations used when encoding or decoding trajectories. It could be: - `none`: no transformations - `fft`: fast Fourier transform - `fft2d`: 2D fast Fourier transform - `haar`: haar wavelet transform - `db2`: DB2 wavelet transform 
  The default value is `fft`.
- `--down_sampling_rate`: type=`float`, argtype=`temporary`.
  Down sampling rate to sample trajectories from all N = K*Kc trajectories. 
  The default value is `1.0`.
- `--feature_dim`: type=`int`, argtype=`static`.
  Feature dimensions that are used in most layers. 
  The default value is `128`.
- `--key_points`: type=`str`, argtype=`static`.
  A list of key time steps to be predicted in the agent model. For example, `'0_6_11'`. 
  The default value is `0_6_11`.
- `--loada` (short for `-la`): type=`str`, argtype=`temporary`.
  Path to load the first-stage agent model. 
  The default value is `null`.
- `--loadb` (short for `-lb`): type=`str`, argtype=`temporary`.
  Path to load the second-stage handler model. 
  The default value is `null`.
- `--preprocess`: type=`str`, argtype=`static`.
  Controls whether to run any pre-process before the model inference. It accepts a 3-bit-like string value (like `'111'`): - The first bit: `MOVE` trajectories to (0, 0); - The second bit: re-`SCALE` trajectories; - The third bit: `ROTATE` trajectories. 
  The default value is `111`.

### First-stage silverballers args

- `--depth`: type=`int`, argtype=`static`.
  Depth of the random noise vector. 
  The default value is `16`.
- `--deterministic`: type=`int`, argtype=`static`.
  Controls if predict trajectories in the deterministic way. 
  The default value is `0`.
- `--loss`: type=`str`, argtype=`temporary`.
  Loss used to train agent models. Canbe `'avgkey'` or `'keyl2'`. 
  The default value is `keyl2`.

### Second-stage silverballers args

- `--points`: type=`int`, argtype=`static`.
  The number of keypoints accepted in the handler model. 
  The default value is `1`.
<!-- DO NOT CHANGE THIS LINE -->

## Thanks

Codes of the Transformers used in this model come from [TensorFlow.org](https://www.tensorflow.org/tutorials/text/transformer);  
Dataset CSV files of ETH-UCY come from [SR-LSTM (CVPR2019) / E-SR-LSTM (TPAMI2020)](https://github.com/zhangpur/SR-LSTM);  
Original dataset annotation files of SDD come from [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/), and its split file comes from [SimAug (ECCV2020)](https://github.com/JunweiLiang/Multiverse);  
The nuScenes dataset from [their home page](https://nuscenes.org/nuscenes);
The Human3.6M dataset from [their home page](http://vision.imar.ro/human3.6m/description.php);  
All contributors of the repository [Vertical](https://github.com/cocoon2wong/Vertical).

## Contact us

Conghao Wong ([@cocoon2wong](https://github.com/cocoon2wong)): conghaowong@icloud.com  
Beihao Xia ([@NorthOcean](https://github.com/NorthOcean)): xbh_hust@hust.edu.cn
