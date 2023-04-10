from mmseg.apis import inference_model, init_model, show_result_pyplot
import argparse
import os
import os.path as osp
import mmcv

def parse_args():
    parser = argparse.ArgumentParser(description=
                                     'Inference with a segmentor')
    parser.add_argument('--config', 
                        default='models/bisenetv2_fcn_4x4_skycityscape_1024x1024.py', 
                        help='model config file path')
    parser.add_argument('--checkpoint', 
                        default='checkpoints/bisenetv2_skycityscape_20230410/iter_400.pth', 
                        help='checkpoint file path')
    parser.add_argument('--device', 
                        default='cuda:0', 
                        help='device used for inference')
    parser.add_argument('--img',
                        default=None, 
                        help='image file to be predicted')
    parser.add_argument('--video',
                        default='demo/src/example.mp4', 
                        help='video to be predicted')
    parser.add_argument('--save_dir',
                        default='demo', 
                        help='image file to be predicted')
    args = parser.parse_args()
    return args


def main(args):

    # define path to save the predict results
    save_dir = args.save_dir

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # predict a single image and show the results
    if args.img:
        # read the image
        img = mmcv.imread(args.img)
        # inference the image
        result = inference_model(model, img)
        # visualize the results in a new window and save the results
        out_file = osp.join(save_dir, 'result.jpg')
        show_result_pyplot(model, img, result, show=True, out_file=out_file, opacity=0.5)
    
    # predict a video and show the results
    if args.video:
        # read the video
        video = mmcv.VideoReader(args.video)
        # create the save directory
        save_dir = osp.join(save_dir, 'result')
        os.makedirs(save_dir, exist_ok=True)
        # inference the video
        for idx, frame in enumerate(video):
            result = inference_model(model, frame)
            out_file = osp.join(save_dir, f'{idx}.jpg')
            show_result_pyplot(model, frame, result, wait_time=0.5, opacity=0.5, show=True, out_file=out_file)

if __name__ == "__main__":
    args = parse_args()
    main(args)