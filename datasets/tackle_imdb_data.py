# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-9-24'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
filter IMDB datasets for training.
"""

import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm
from datetime import datetime
from scipy.io import loadmat


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))
    
    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1


def get_meta(mat_path, db):
    meta = loadmat(mat_path)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]
    
    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str,
                        default="/home/data/CVAR-B/study/projects/face_properties/age_estimation/datasets/IMDB/filtered_imdb_crop",
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="imdb",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=64,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    output_path = args.output
    image_output_path = os.path.join(output_path, 'resized_64')
    if not os.path.exists(image_output_path):
        os.makedirs(image_output_path)
    db = args.db
    img_size = args.img_size
    min_score = args.min_score
    
    root_path = '/home/data/CVAR-B/study/datasets/public_datasets/IMDB/imdb_crop'
    mat_path = '/home/data/CVAR-B/study/datasets/public_datasets/IMDB/imdb.mat'
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    
    out_genders = []
    out_ages = []
    out_imgs = []
    
    train_csv = open('train.csv', 'w')
    train_csv.write("{},{}\n".format('name', 'age'))
    count = 0
    total_count = 0
    for i in tqdm(range(len(face_score))):
        total_count = total_count + 1
        if face_score[i] < min_score:
            continue
        
        if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
            continue
        
        if ~(0 <= age[i] <= 100):
            continue
        
        if np.isnan(gender[i]):
            continue
        
        out_genders.append(int(gender[i]))
        out_ages.append(age[i])
        img = cv2.imread(os.path.join(root_path, str(full_path[i][0])))
        img = cv2.resize(img, (img_size, img_size))
        # out_imgs.append(cv2.resize(img, (img_size, img_size)))
        temp_name = str(full_path[i][0]).replace('/', '_')
        flag = cv2.imwrite(os.path.join(image_output_path, temp_name), img)
        if flag:
            train_csv.write('{},{}\n'.format(temp_name, age[i]))
            count = count + 1
            if count % 1000 == 0:
                print("\t{}/{} processed...".format(count, total_count))
    
    print("\t{}/{} processed...".format(count, total_count))
    np.savez(output_path, image=np.array(out_imgs), gender=np.array(out_genders), age=np.array(out_ages),
             img_size=img_size)


if __name__ == '__main__':
    main()
