# Real-time Sky Segmentation

This project aims to develop a lightweight model to segment the sky in images, utilizing OpenMMLab's open-source packages.  

## Environment & Dataset

The project primarily depends on the latest versions of MMSegmentation, MMEngine, and MMDeploy. The runtime environment is based on Python 3.9, PyTorch 1.12, CUDA 11.3, etc. The [Dockerfile](docker/Dockerfile) provides a comprehensive list of packages used in this project (please note that the build and runtime of the Docker image have not been tested due to local storage limitations).

[Cityscapes dataset](https://www.cityscapes-dataset.com/) is used for training the model. It should be downloaded and placed in the following directory structure:

```
RealtimeSkySeg
├── data
│   ├──datasets
│   │   ├── cityscapes
│   │   │   ├── leftImg8bit
│   │   │   │   ├── test
│   │   │   │   ├── train
│   │   │   │   ├── val
│   │   │   ├── gtFine
│   │   │   │   ├── test
│   │   │   │   ├── train
│   │   │   │   ├── val
```
Images and annotations should be preprocessed using the [cityscapes_formater.py](tools/cityscape_formater.py) script, which reduces the labels to two classes: `not_sky` and `sky`.

The newly formated dataset [SkyCityscapes](data/configs/SkyCityscapesDataset.py) is defined as a custom dataset in MMSegmentation.

## Model Training

The model used for this project is BiSeNetV2.

BiSeNetV2 is recognized as a modern baseline model for real-time semantic segmentation. It is designed to achieve high accuracy while also being efficient enough for real-time applications, making it a popular choice for tasks such as autonomous driving, robotics, and surveillance.

References:
- [BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/abs/2004.02147)

To train the model with our SkyCityscapes dataset, run the following command:
```
python tools/train_segmentor.py models/bisenetv2_fcn_4x4_skycityscape_1024x1024.py --work-dir checkpoints/bisenetv2_skycityscapes --amp
```
The current configuration fine tune the pre-trained model for 2000 iterations by default, with a starting learning rate of 0.01. More details can be found in the [config file](models/bisenetv2_fcn_4x4_skycityscape_1024x1024.py).

Current training results:
| model   |  dataset   | sky test IoU  | sky test accuracy
| --------| ---------- | -------- | ------ |
| BiSeNetV2-4xb4    | SkyCityscapes, random croped at 1024 x 1024    | 71.4 | 90.4

More tuning on training hyperparameters or the utilization of other models may achieve better results. Exploration is limited due to time constraints.

## Model Inference
Perform model inference with MMSegmentation APIs. For images, use the following command:
```
python tools/inference_base.py --config models/bisenetv2_fcn_4x4_skycityscape_1024x1024.py --checkpoints checkpoints/LATEST.pt --img demo/src/example.png --save_dir demo 
```
For videos, use this command:
```
python tools/inference_base.py --config models/bisenetv2_fcn_4x4_skycityscape_1024x1024.py --checkpoints checkpoints/LATEST.pt --video demo/src/example.mp4 --save_dir demo 
```
These commands use the MMSegmentation APIs to perform model inference on either images or videos. The results will be saved in the specified save_dir. Note that the performance of the model may vary depending on the input data and hardware resources. 

### Real-time inference
For real-time applications, it's important to optimize the model further to reduce latency and improve throughput. One approach to achieve this is by converting the trained PyTorch model to ONNX and then to TensorRT.
