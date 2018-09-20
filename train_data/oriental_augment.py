#!/usr/bin/python
#encoding: utf-8
import os
import glob
import cv2
import numpy as np
import random
from time import time
import sctool
from easydict import EasyDict as edict
import copy



''' 
  Augment dataset by oriental encoding:
  
  Suppose the VOC-format dataset has been annotated as BBOX on the image I with rotation angle A
  This script will generate:
    1. rotated image rot(I,A) with the annotation BBOX
    2. raw image and objs with name K inside, since K is rotated in raw image,
       we generate the bouding box of K by back mapping BBOX to BBOX_BIGGER, 
       and we encode the rotation angle as the left/right diagnal orientation of BBOX_BIGGER
       either its left or right diagonal is further encoded in new label KL or KR
    3. random further rotation dA applied on I, and save it as 2

'''

img_file_list_src = glob.glob("/home/hddl/dockerv0/second_candy/train_data/JPEGImages/*.jpeg")
xml_file_path_src = "./Annotations"

img_file_list_dst = "./JPEGImages2/"
xml_file_path_dst = "./Annotations2"


if not os.path.exists(img_file_list_dst): os.makedirs(img_file_list_dst)
if not os.path.exists(xml_file_path_dst): os.makedirs(xml_file_path_dst)

    
def filename_add_suffix(filepath, suffix):
    base, ext = os.path.splitext(os.path.basename(filepath))
    return os.path.join(img_file_list_dst, "{}_{}{}".format(base, suffix, ext))


def make_corrected(image, objs, imgfilepath, A):
    image0, mapper = sctool.rotateImage(image, A)          # now image0 has rotation A-A = 0
    fname = filename_add_suffix(imgfilepath, "raw")
    cv2.imwrite(fname, image0)
    sctool.save_voc_annotation(fname, 
                               objs,               # bbox can be applied directly
                               rotation_degree=0, xmlpath=xml_file_path_dst)
    return image0



def make_rotated(image0, image, objs, imgfilepath, A, dA = 0):
    image1, _ = sctool.rotateImage(image, dA)  # now image1 has rotation A-dA
                                                    # so rotateImage(image1, A-dA) will get image0
    _, mapper = sctool.rotateImage(image1, A-dA)
    
    fname = filename_add_suffix(imgfilepath, "r{}".format(int(A-dA)))
    cv2.imwrite(fname, image1)
    
    objs = copy.deepcopy(objs)
    
    # now rotate BBOXes accordingly
    for obj in objs:
        x0 = obj.bndbox.xmin
        y0 = obj.bndbox.ymin
        x1 = obj.bndbox.xmax
        y1 = obj.bndbox.ymax
        pts = mapper(np.array([[(x0+x1)/2,y0],[(x0+x1)/2,y1]]).T, False).T
        
        # the bounding box's diagonal was used to encode A-dA
        obj.bndbox.xmin, obj.bndbox.ymin = np.amin(pts, 0)
        obj.bndbox.xmax, obj.bndbox.ymax = np.amax(pts, 0)
        
        # the box's name (class) was used to encode which diagonal is (A-dA)
        if (A-dA >= 0):
            obj.name += "L"
        else:
            obj.name += "R"
    
    sctool.save_voc_annotation(fname, 
                               objs, 
                               rotation_degree=0, xmlpath=xml_file_path_dst)    
    



for ind, fimg in enumerate(img_file_list_src):
    print("{:.1%}".format(float(ind)/len(img_file_list_src)), fimg)
    
    objs, rotate_angle = sctool.load_voc_annotation(fimg, xmlpath=xml_file_path_src)    
    
    image = cv2.imread(fimg)
    
    # corrected image
    image0 = make_corrected(image, objs, fimg, rotate_angle)
    
    make_rotated(image0, image, objs, fimg, rotate_angle, 0)
    
    continue
    # now randomly select dA so rotate_angle-dA is within (-90,90)
    # or dA is within (A-90,A+90) but since we rotate from A by dA, dA is also need to be in (-90,90)
    for i in np.random.choice(180, 50):
        dA = rotate_angle - 90 + i
        if dA <= -90 and dA <= 90:
            make_rotated(image0, image, objs, fimg, rotate_angle, dA)
    

