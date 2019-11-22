# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '2019-10-18'
__email__ = 'kohou.wang@cloudminds.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
inference one single image for leixuejiao.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import copy
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from datasets.read_imdb_data import IMDBDatasets
from datasets.read_megaasian_data import MegaAgeAsianDatasets
from datasets.read_face_age_data import FaceAgeDatasets
from SSR_models.SSR_Net_model import SSRNet
from SSR_models.ssrnet_hans import ssrnet_hans
from torch.utils.tensorboard import SummaryWriter
import argparse
import time

import torch.nn as nn

import numpy as np
import torch
from torchvision import transforms as T
import cv2
from torch.nn import init
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inference_single_image(model_, image_path_, input_size_=64):
    image_ = cv2.imread(image_path_)
    if image_.shape[0] != input_size_:
        image_ = cv2.resize(image_, (input_size_, input_size_))
    start_time_ = time.time()
    image_ = T.Compose([
        T.ToPILImage(),
        T.Resize((input_size_, input_size_)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(image_)
    
    image_ = image_[np.newaxis,]
    image_ = image_.cuda()
    results_ = model_(image_)
    return results_,  time.time() - start_time_


if __name__ == "__main__":
    image_file_path = "/home/CVAR-B/study/projects/face_properties/age_estimation/datasets/megaage_asion/megaage_asian/megaage_asian/test"
    model_file = "/home/CVAR-B/study/projects/face_properties/age_estimation/trained_models/SSR_Net_MegaAge_Asian/model_Adam_MSELoss_LRDecay_weightDecay0.0001_batch50_lr0.0005_epoch90_64x64.pth"
    
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
    
    if os.path.isfile(image_file_path):
        age_, cost_time = inference_single_image(inference_model, image_file_path)
        print("used {} s in total.".format(cost_time))
        print(age_.data, age_.shape)
        # print(embeddings.data, embeddings.shape)
    elif os.path.isdir(image_file_path):
        results_list = []
        for image in os.listdir(image_file_path):
            age_, _ = inference_single_image(inference_model, os.path.join(image_file_path, image))
            
            results_list.append(age_.tolist()[0])
            # print("quality:\t{}\t, image:\t{}".format(torch.nn.functional.softmax(logits), image))
            print("age:\t{}\t, image:\t{}".format(age_.tolist()[0], image))
        import pandas as pd
        
        pd_result = pd.DataFrame(results_list)
        print(pd_result.describe())
        print(pd_result[0].value_counts())
