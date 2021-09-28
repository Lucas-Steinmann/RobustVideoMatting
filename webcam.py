#! /usr/bin/env python3
import argparse
import logging
from typing import Optional

import numpy as np
import torch
import cv2
import pyvirtualcam
from torchvision.transforms import ToTensor

from model import MattingNetwork


def webcam_setup_and_loop(
        backbone: str = 'mobilenetv3',
        model_weights: str = 'rvm_mobilenetv3.pth',
        input_filename: str = "/dev/video0",
        background: str = 'green',
        pref_width: int = 1280,
        pref_height: int = 720,
        pref_fps: int = 30,
        downsample_ratio: float = 0.25,
        device: Optional[str] = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'Using device {device}.')

    # Initialize model, move it to the desired device
    # and load the trained weights
    model = MattingNetwork(backbone).eval().to(device)
    model.load_state_dict(torch.load(model_weights))

    # Use this transform to convert images and video frames
    # from (H, W, C) np.arrays to (C, H, W) torch.tensors
    transform = ToTensor()

    # Initialize video input
    reader = cv2.VideoCapture(input_filename)
    if not reader.isOpened():
        raise RuntimeError('Could not open video source')
    # Try set preferred input size
    reader.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
    reader.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
    reader.set(cv2.CAP_PROP_FPS, pref_fps)
    # Get actual input size
    width = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    logging.info(f"Using input width: {width}, height: {height}, FPS: {fps}.")

    # Initialize background
    if background == 'green':
        # Green background.
        bgr = torch.tensor([.47, 1, .6]).view(3, 1, 1).to(device)
    else:
        # Image background
        bgr = cv2.imread(background)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        bgr = transform(cv2.resize(bgr, (width, height))).to(device)

    # Initial recurrent states.
    rec = [None] * 4

    with torch.no_grad():
        # Initialize virtual camera with same size and frame rate
        with pyvirtualcam.Camera(width=width, height=height, fps=fps) as cam:
            while True:
                ret, src = reader.read()
                src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)              # BGR -> RGB
                src = transform(src).unsqueeze(0)                       # Convert to tensor of shape (1, C, H, W)
                src = src.to(device)                                        # and move to correct device
                fgr, pha, *rec = model(src, *rec, downsample_ratio)     # Cycle the recurrent states.
                com = fgr * pha + bgr * (1 - pha)                       # Composite to background.
                com = com[0].permute(1, 2, 0).cpu().numpy()             # Convert back to array of shape (H, W, C)
                com = (255 * com).astype(np.uint8)                      # float to uint8 and set range to [0, 255]
                cam.send(com)                                           # Send frame to virtual camera
                cam.sleep_until_next_frame()


def set_log_level(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_level)
    logging.basicConfig(level=numeric_level)


def parse_args():
    parser = argparse.ArgumentParser('Robust webcam video matting.',
                                     description="""
Reads frames from an input stream via opencv's VideoCapture.
Then passes the frames through the robust matting network.
The networks mask is used to compose the frames with the 
desired background, green by default, but can be changed 
to a background image using the -b/--background flag.
And finally streams the camera to a new video device 
using pyvirtualcam.
""")
    parser.add_argument('-i', '--input', default='/dev/video0', type=str, required=False,
                        help='The input stream, either a video filename or capturing device.')
    parser.add_argument('-b', '--background', default='green', type=str, required=False,
                        help='Either green, for a green background or path to a background image.')
    parser.add_argument('--backbone', default='mobilenetv3', type=str, required=False,
                        help='The type of CNN backbone, which the model should use. '
                             'Changes quality and performance. If changed, also make sure you point'
                             '--model-weights to the matching model weights.',
                        choices=('mobilenetv3', 'resnet50'))
    parser.add_argument('--model-weights', default='rvm_mobilenetv3.pth', type=str, required=False,
                        help='The path to the trained model weights. Must match the backbone.')
    parser.add_argument('-w', '--width', default=1280, type=int, required=False,
                        help='The preferred width of the input, which will be equal to the output width.')
    parser.add_argument('--height', default=720, type=int, required=False,
                        help='The preferred height of the input, which will be equal to the output height.')
    parser.add_argument('-f', '--fps', default=30, type=int, required=False,
                        help='The preferred frame rate of the input, which will be equal to the output frame rate.')
    parser.add_argument('-r', '--downsample-ratio', default=0.25, type=float, required=False,
                        help='The downsample ratio, which should be used before forwarding the frame with the CNN.'
                             'Should be decreased, if computational resources are a problem.')
    parser.add_argument('-d', '--device', default=None, type=str, required=False,
                        help='The device on which to execute the neural network.'
                             'If not set, a CUDA device will be tried.'
                             'If not available, CPU is used as fallback.')
    parser.add_argument('--log-level', default='INFO', type=str, required=False,
                        help='Log level to use. Must match the name of a '
                             'log level in pythons builtin logging framework.')
    parsed_args = parser.parse_args()
    set_log_level(parsed_args.log_level)
    return parsed_args


if __name__ == '__main__':
    args = parse_args()
    webcam_setup_and_loop(backbone=args.backbone,
                          model_weights=args.model_weights,
                          background=args.background,
                          pref_width=args.width,
                          pref_height=args.height,
                          pref_fps=args.fps,
                          downsample_ratio=args.downsample_ratio,
                          device=args.device)
