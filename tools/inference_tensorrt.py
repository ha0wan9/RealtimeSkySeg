import argparse
import os
import os.path as osp
import cv2
import numpy as np
from mmdeploy import DeployBackend

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a model using tensor RT.")
    parser.add_argument("model_path",
                        help="path to the .trt model file.")
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


def preprocess(img, input_shape):
    """
    Preprocess the input image for inference.

    Args:
        img (numpy.ndarray): The input image in BGR format.
        input_shape (tuple): The target input shape for the model (width, height).

    Returns:
        numpy.ndarray: The preprocessed image tensor.
    """
    # Resize the input image to the target input shape for the model
    img_resized = cv2.resize(img, input_shape)
    
    # Normalize the image by dividing each pixel value by 255.0
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Transpose the image dimensions to (channels, height, width) and add a batch dimension
    img_tensor = np.transpose(img_normalized, (2, 0, 1))[np.newaxis, :]
    
    return img_tensor

def postprocess(output):
    """
    Postprocess the output tensor to generate a segmentation map.

    Args:
        output (list): The output tensor from the model.

    Returns:
        numpy.ndarray: The segmentation map.
    """
    # Find the class with the highest probability for each pixel
    segmentation_map = np.argmax(output[0], axis=0)
    
    return segmentation_map

def visualize(frame, segmentation_map):
    """
    Visualize the segmentation map by overlaying it on the input frame.

    Args:
        frame (numpy.ndarray): The input frame in BGR format.
        segmentation_map (numpy.ndarray): The segmentation map.

    Returns:
        numpy.ndarray: The visualization frame with the segmentation map overlaid.
    """
    # Define a color map for each class
    color_map = np.array([[0, 0, 0], [70, 130, 180]])
    
    # Map the segmentation map to the color map
    colored_map = color_map[segmentation_map]
    
    # Overlay the colored map on the input frame with 50% transparency
    vis_frame = cv2.addWeighted(frame, 0.5, colored_map, 0.5, 0)
    
    return vis_frame

def main(args):
    """
    Perform real-time inference using a TensorRT model on an input image or video.

    Args:
        args: Parsed command-line arguments.
    """
    # Set the output directory for saving the results
    save_dir = args.save_dir

    # Initialize the TensorRT backend with the exported TensorRT model
    backend = DeployBackend('tensorrt', args.model_path)

    # Define the input shape for the model
    input_shape = (1024, 1024)

    # Perform inference on a single image
    if args.img:
        # Read the input image
        img = cv2.imread(args.img)
        
        # Preprocess the input image
        input_data = preprocess(img, input_shape)
        
        # Perform inference with the TensorRT backend
        output = backend.inference(inputs=[input_data])
        
        # Postprocess the model output to generate the segmentation map
        segmentation_map = postprocess(output)
        
        # Visualize the segmentation map by overlaying it on the input image
        vis_frame = visualize(img, segmentation_map)
        
        # Save the visualization result to a file
        cv2.imwrite(osp.join(save_dir, 'result.jpg'), vis_frame)

    # Perform inference on a video
    if args.video:
        # Open the input video file
        cap = cv2.VideoCapture(args.video)
        
        # Create a directory for saving the video frames with segmentation maps overlaid
        save_dir = osp.join(save_dir, 'result')
        os.makedirs(save_dir, exist_ok=True)

        idx = 0
        while cap.isOpened():
            # Read a frame from the video
            ret, frame = cap.read()
            
            # Break the loop if we have reached the end of the video
            if not ret:
                break

            # Preprocess the input frame
            input_data = preprocess(frame, input_shape)
            
            # Perform inference with the TensorRT backend
            output = backend.inference(inputs=[input_data])
            
            # Postprocess the model output to generate the segmentation map

            segmentation_map = postprocess(output)

            # Visualize the segmentation map by overlaying it on the input frame
            vis_frame = visualize(frame, segmentation_map)

            # Save the visualization result to a file
            cv2.imwrite(osp.join(save_dir, f'{idx}.jpg'), vis_frame)

            # Display the visualization result in a window
            cv2.imshow('Real-time Segmentation', vis_frame)
            idx += 1

            # Break the loop if the 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the video capture object and close all windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
