import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
from scipy.ndimage.filters import gaussian_filter
import random

def dataAugmentation(img_path, bbox_path, angle=0):
    image = Image.open(img_path)
    # convert image to numpy array
    img_array = np.asarray(image)

    img_rot = np.rot90(img_array, axes=(0, 1))
    #img_rot[:, :, 0] = gaussian_filter(img_rot[:, :, 0], sigma=random.uniform(0, 0.1))
    #img_rot[:, :, 1] = gaussian_filter(img_rot[:, :, 1], sigma=random.uniform(0, 0.1))
    #img_rot[:, :, 2] = gaussian_filter(img_rot[:, :, 2], sigma=random.uniform(0, 0.1))

    im = Image.fromarray(np.uint8(img_rot))
    if angle == '180':
        img_path_new = f"{img_path.split('.')[0][:-2]}{angle}.jpg"
        bbox_path_new  = f".{bbox_path.split('.')[1][:-2]}{angle}.txt"
    elif angle == '270':
        img_path_new = f"{img_path.split('.')[0][:-3]}{angle}.jpg"
        bbox_path_new  = f".{bbox_path.split('.')[1][:-3]}{angle}.txt"
    else:
        img_path_new = f"{img_path.split('.')[0]}_{angle}.jpg"
        bbox_path_new  = f".{bbox_path.split('.')[1]}_{angle}.txt"
    im.save(img_path_new)

    with open(bbox_path_new, 'w') as f_w:
        with open(bbox_path, 'r') as f:
            bbox = f.readlines()
            bbox = [bb.replace('\n', '') for bb in bbox]
            for bb in bbox:
                new_coordinates = []
                coordinates = bb.split(' ')

                f_w.write(f"{coordinates[0]} {float(coordinates[2])} {1 - float(coordinates[1])} {coordinates[4]} {coordinates[3]}\n")
    return img_path_new, bbox_path_new

def dataAugmentationBlack(img_path, bbox_path):
    img_path_new = f"{img_path.split('.')[0]}_black.jpg"
    bbox_path_new = f".{bbox_path.split('.')[1]}_black.txt"
    img = mpimg.imread(img_path)
    x, y, _ = img.shape
    paths_base = next(os.walk(f"./NWPU/negative image set/"))[2]
    new_img = mpimg.imread(f"./NWPU/negative image set/{random.choice(paths_base)}")

    while x > new_img.shape[0]:
        new_img = np.concatenate((new_img, new_img), axis=0)[:x, :, :]

    new_img = new_img[:x, :, :]
    while y > new_img.shape[1]:
        new_img = np.concatenate((new_img, new_img), axis=1)[:, :y, :]
    new_img = new_img[:, :y, :]


    with open(bbox_path_new, 'w') as f_w:
        with open(bbox_path, 'r') as f:
            bbox = f.readlines()
            [f_w.write(bb) for bb in bbox]
            bbox = [bb.replace('\n', '') for bb in bbox]
            for bb in bbox:
                new_coordinates = []
                l = bb.split(' ')
                l[1] = int(float(l[1]) * y) - int(float(l[3]) * y / 2)
                l[2] = int(float(l[2]) * x) - int(float(l[4]) * x / 2)
                l[3] = l[1] + int(float(l[3]) * y)
                l[4] = l[2] + int(float(l[4]) * x)
                new_img[l[2]:l[4], l[1]:l[3], :] = img[l[2]:l[4], l[1]:l[3], :]

    new_img[:, :, 0] = gaussian_filter(new_img[:, :, 0], sigma=random.uniform(0, 0.5))
    new_img[:, :, 1] = gaussian_filter(new_img[:, :, 1], sigma=random.uniform(0, 0.5))
    new_img[:, :, 2] = gaussian_filter(new_img[:, :, 2], sigma=random.uniform(0, 0.5))
    new_img = Image.fromarray(np.uint8(new_img))
    new_img.save(img_path_new)
    return img_path_new, bbox_path_new


novel_classes = ["airplane","baseball-diamond","tennis-court"]
path_novel = []
path_bbox = []
for novel in novel_classes:
    path_txt = f"./nwpulist/box_10shot_{novel}_train.txt"
    with open(path_txt, 'r') as f:
        path_novel.append(f.readlines())

    print(next(os.walk(f"./labels_1c/{novel}_10shot/"))[2])
    path_bbox.append([f"./labels_1c/{novel}_10shot/{i}" for i in next(os.walk(f"./labels_1c/{novel}_10shot/"))[2]])

path_novel = [item.replace('\n', '') for sublist in path_novel for item in sublist]
path_novel = list(path_novel)
path_bbox = [item.replace('\n', '') for sublist in path_bbox for item in sublist]
path_bbox = list(path_bbox)
#path_bbox = [(f"./training/annotations/{p.split('/')[-1].replace('.jpg', '.txt')}") for p in path_novel]

#print("IMAGE")
#print(path_novel)
#print("BBOX")
#print(path_bbox)

path_img_90 = []
path_bbox_90 = []
#for novel, bbox in zip(path_novel, path_bbox):
for novel in path_novel:
    for bbox in path_bbox:
        if novel[-17:-4] == bbox[-17:-4]:
            img_new, bbox_new = dataAugmentation(novel, bbox, '90')
            path_img_90.append(img_new)
            path_bbox_90.append(bbox_new)

path_img_180 = []
path_bbox_180 = []
for novel, bbox in zip(path_img_90, path_bbox_90):
    img_new, bbox_new = dataAugmentation(novel, bbox, '180')
    path_img_180.append(img_new)
    path_bbox_180.append(bbox_new)

path_img_270 = []
path_bbox_270 = []
for novel, bbox in zip(path_img_180, path_bbox_180):
    img_new, bbox_new = dataAugmentation(novel, bbox, '270')
    path_img_270.append(img_new)
    path_bbox_270.append(bbox_new)

path_img_black = []
path_bbox_black = []
for novel in path_novel:
    for bbox in path_bbox:
        if novel[-17:-4] == bbox[-17:-4]:
            img_new, bbox_new = dataAugmentationBlack(novel, bbox)
            path_img_black.append(img_new)
            path_bbox_black.append(bbox_new)

path_img_90_black = []
path_bbox_90_black = []
for novel, bbox in zip(path_img_90, path_bbox_90):
    img_new, bbox_new = dataAugmentationBlack(novel, bbox)
    path_img_90_black.append(img_new)
    path_bbox_90_black.append(bbox_new)

path_img_180_black = []
path_bbox_180_black = []
for novel, bbox in zip(path_img_180, path_bbox_180):
    img_new, bbox_new = dataAugmentationBlack(novel, bbox)
    path_img_180_black.append(img_new)
    path_bbox_180_black.append(bbox_new)

path_img_270_black = []
path_bbox_270_black = []
for novel, bbox in zip(path_img_270, path_bbox_270):
    img_new, bbox_new = dataAugmentationBlack(novel, bbox)
    path_img_270_black.append(img_new)
    path_bbox_270_black.append(bbox_new)


    


