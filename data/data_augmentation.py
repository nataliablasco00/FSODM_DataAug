import os
import sys
import numpy as np
from PIL import Image

def dataAugmentation(img_path, bbox_path, angle=0):
    image = Image.open(img_path)
    # convert image to numpy array
    img_array = np.asarray(image)
    img_rot = np.rot90(img_array)
    im = Image.fromarray(np.uint8(img_rot*255))
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
                
                # x1_new = img_rot.shape[1]-coordinates[1]
                # y1_new = coordinates[0]

                # x2_new = img_rot.shape[1] - coordinates[3]
                # y2_new = coordinates[2]

                f_w.write(f"{img_rot.shape[1]-float(coordinates[1])} {coordinates[0]} {img_rot.shape[1] - float(coordinates[3])} {coordinates[2]} {coordinates[4]}\n")
    return img_path_new, bbox_path_new

novel_classes = ["airplane","baseball-diamond","tennis-court"]
path_novel = []
for novel in novel_classes:
    path_txt = f"./nwpulist/box_10shot_{novel}_train.txt"
    with open(path_txt, 'r') as f:
        path_novel.append(f.readlines())

path_novel = [item.replace('\n', '') for sublist in path_novel for item in sublist]
path_novel = list(set(path_novel))
path_bbox = [(f"./training/annotations/{p.split('/')[-1].replace('.jpg', '.txt')}") for p in path_novel]

path_img_90 = []
path_bbox_90 = []
for novel, bbox in zip(path_novel, path_bbox):
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

print(path_img_270, path_bbox_270)


    


