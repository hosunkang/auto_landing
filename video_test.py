import argparse
import os

import torch
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw

import model.yolov3
import utils.datasets
import utils.utils

#### About realsense webcam
import webcam 
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--pretrained_weights", type=str, default="weights/yolov3_voc.pth",
                    help="path to pretrained weights file")
parser.add_argument("--image_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터셋 설정값을 가져오기
data_config = utils.utils.parse_data_config(args.data_config)
num_classes = int(data_config['classes'])
class_names = utils.utils.load_classes(data_config['names'])

# 모델 준비하기
model = model.yolov3.YOLOv3(args.image_size, num_classes).to(device)
print(model)
if args.pretrained_weights.endswith('.pth'):
    model.load_state_dict(torch.load(args.pretrained_weights))
else:
    model.load_darknet_weights(args.pretrained_weights)
model.eval() 

rs = webcam.realsense()

while 1:
    color_image = rs.get_cam()
    image = utils.datasets.ImageResize.imgresize(Image.fromarray(color_image), args.image_size)
    print(type(image))

    with torch.no_grad():
        image = image.to(device)
        prediction = model(image)
        prediction = utils.utils.non_max_suppression(prediction, args.conf_thres, args.nms_thres)

    cv2.imshow('image',image)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        break

