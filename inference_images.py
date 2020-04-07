# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-10-22'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
inference one single image or many images of a directory using pretrained SSRNet.
"""

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from SSR_models.SSR_Net_model import SSRNet
import argparse
import time

import numpy as np
import torch
from torchvision import transforms as T
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inference_single_image(model_, image_path_, input_size_=64):
    image_ = cv2.imread(image_path_)
    start_time_ = time.time()
    image_ = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size_, input_size_)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image_)
    
    image_ = image_[np.newaxis, ]
    image_ = image_.to(device)
    results_ = model_(image_)
    return results_,  time.time() - start_time_


if __name__ == "__main__":
    image_file_path = "../../datasets/megaage_asion/megaage_asian/megaage_asian/test/13.jpg"
    model_file = "./pretrained_model/model_Adam_MSELoss_LRDecay_weightDecay0.0001_batch50_lr0.0005_epoch90_64x64.pth"
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="image to be processed, dir or a single image.")
    parser.add_argument("--graph", help="graph/model to be executed")
    
    args = parser.parse_args()
    
    if args.graph:
        model_file = args.graph
    if args.image:
        image_file_path = args.image
    
    input_size = 64
    
    inference_model = SSRNet()
    loaded_model = torch.load(model_file)
    inference_model.load_state_dict(loaded_model['state_dict'])
    inference_model = inference_model.to(device)
    inference_model.eval()
    
    if os.path.isfile(image_file_path):  # inference a single image
        age_, cost_time = inference_single_image(inference_model, image_file_path)
        print("age:\t{}, used {} s in total.".format(age_[0], cost_time))
    elif os.path.isdir(image_file_path):  # a directory containing many images, inference them all!
        results_list = []
        for image in os.listdir(image_file_path):
            age_, _ = inference_single_image(inference_model, os.path.join(image_file_path, image))
            
            results_list.append(age_.tolist()[0])
            print("age:\t{}\t, image:\t{}".format(age_.tolist()[0], image))
        import pandas as pd
        
        # just a glimpse of the predicted results.
        pd_result = pd.DataFrame(results_list)
        print(pd_result.describe())
        print(pd_result[0].value_counts())
