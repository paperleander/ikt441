import os
import sys
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from tqdm import tqdm 


images_path = "../data/fashion-product-images-small/images"
styles_path = "../data/fashion-product-images-small/styles_pps.csv"
dataset_path = "../data/fashion-product-images-small/"
new_path = "../data/fashion-shoes" 


def crop(img):
    img_shape = img.shape[:2]
    max_len = max(img_shape) # 80
    min_len = min(img_shape) # 60
    index_max = np.argmax(img_shape) # 0
    index_min = np.argmin(img_shape) # 1
    
    # find min and max of crop
    min_d = int((max_len - min_len)/2)
    max_d = max_len - min_d
    
    if index_min == 1:
        return img[min_d:max_d, :]
    
    return img[:, min_d:max_d]


def resize_img(img, res):
#     scale_percent = 48 # percent of original size
#     width = int(img.shape[1] * scale_percent / 100)
#     height = int(img.shape[0] * scale_percent / 100)
    dim = (res, res)
    # resize image
    resized = cv2.resize(img, dim)
    return resized

def process_and_save(img_path, save_path, res):
    img = cv2.imread(img_path)
    img = crop(img)
    img = resize_img(img, res)
    cv2.imwrite(save_path, img)    

def create_dataset(new_path_res, res):
    # extract shoes from csv
    df = pd.read_csv(styles_path, error_bad_lines=False)
    df['image'] = df.apply(lambda row: str(row['id']) + ".jpg", axis=1)
    shoes_list = df[df["subCategory"] == "Shoes"]["image"].tolist()

    # process images
    for shoe in tqdm(shoes_list, desc='copying images'):
#         from_path = os.path.join(images_path, shoe)
#         to_path = os.path.join(new_path, shoe)
#         print("copy {}".format(shoe))
#         shutil.copy(from_path, to_path)
        img_path = os.path.join(images_path, shoe)
        save_path = os.path.join(new_path_res, shoe)
        process_and_save(img_path, save_path, res)

if __name__ == "__main__":
        
    resolutions = [28, 56]
    
    # print info
    print("current working dir: {}".format(os.getcwd()))
    print("contents in data/:\n{}".format(os.listdir("../data")))
    print("images path: {}".format(images_path))
    print("styles path: {}".format(styles_path))
    print("dataset path: {}".format(dataset_path))
    print("new path: {}".format(new_path))
    
    # check if zip has been extracted
    if not os.path.exists(dataset_path):
        print("[WARN] please unzip the zip file")
        sys.exit()

    # check for pps styles
    if not os.path.exists(styles_path):
        print("[WARN] use the preprocessed styles file styles_pps.csv")
    
    for res in resolutions:
        new_path_res = "{}-{}".format(new_path, str(res))
        
        # create new folder
        if os.path.exists(new_path_res):
            print("[INFO] folder exists. skipping.")
            continue
        
        print("[INFO] creating folder: {}".format(new_path))
        os.mkdir(new_path_res)
        try:
            create_dataset(new_path_res, res)    
        except Exception as e:
            print("[ERROR] {}".format(e))
            print("[INFO] removing folders")
            shutil.rmtree(new_path_res)
                
    sys.exit()

        
   
