from os import listdir
import os
from os.path import isfile, join
import shutil


processed_path = "./DL_test"
base_path = "./DL_data"
val_txt_path = base_path + "/val.txt"

with open(val_txt_path) as f:
    lines = f.readlines()
    for i, line in enumerate(lines):
        img_name, label = line.split(" ")

        label = label.replace("\n","")

        img_dir_path = processed_path + "/" + label
        img_src_path = base_path + "/test_images/" + img_name
        img_dst_path = img_dir_path + "/" + img_name
        if not os.path.isdir(img_dir_path):
            os.mkdir(img_dir_path)

        shutil.copyfile(img_src_path, img_dst_path)

# for i in range(0, 1000):
#     img_dir_path = processed_path + "/" + str(i)
#     if not os.path.isdir(img_dir_path):
#         os.mkdir(img_dir_path)


