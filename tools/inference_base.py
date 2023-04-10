from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv

import torch
print(torch.__version__)


config_file = 'models/bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py'
checkpoint_file = 'models/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')
# test a single image and show the results
img = 'demo/example.png'  # or img = mmcv.imread(img), which will only load it once
result = inference_model(model, img)
# visualize the results in a new window
show_result_pyplot(model, img, result, show=True)
# or save the visualization results to image files
# you can change the opacity of the painted segmentation map in (0, 1].
show_result_pyplot(model, img, result, show=True, out_file='result.jpg', opacity=0.5)
# test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#    result = inference_segmentor(model, frame)
#    show_result_pyplot(model, result, wait_time=1)