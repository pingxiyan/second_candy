#!/usr/bin/python
# encoding: utf-8
from __future__ import print_function
import os
import glob
import cv2
import numpy as np
import random
from time import time
import sctool
from easydict import EasyDict as edict
import copy
import sys

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

img_file_list_src = glob.glob("/home/hddl/dockerv0/data/voc/VOCdevkit/tower/JPEGImages/*.jpg")
xml_file_path_src = "/home/hddl/dockerv0/data/voc/VOCdevkit/tower/Annotations"

img_file_list_dst = "/home/hddl/dockerv0/data/voc/VOCdevkit/VOC2019/JPEGImages"
xml_file_path_dst = "/home/hddl/dockerv0/data/voc/VOCdevkit/VOC2019/Annotations"
list_file_path_dst = "/home/hddl/dockerv0/data/voc/VOCdevkit/VOC2019/ImageSets/Main"


if not os.path.exists(img_file_list_dst): os.makedirs(img_file_list_dst)
if not os.path.exists(xml_file_path_dst): os.makedirs(xml_file_path_dst)
if not os.path.exists(list_file_path_dst): os.makedirs(list_file_path_dst)

def filename_add_suffix(filepath, suffix):
    base, ext = os.path.splitext(os.path.basename(filepath))
    return os.path.join(img_file_list_dst, "{}_{}{}".format(base, suffix, ext))

list_fnames = []
cat_counter ={"c":0, "cR":0, "cL":0}

def make_corrected(imageA, objs, imgfilepath, A):
    image0, mapperAto0 = sctool.rotateImage(imageA, A)  # now image0 has rotation A-A = 0
    fname = filename_add_suffix(imgfilepath, "raw")
    cv2.imwrite(fname, image0)
    list_fnames.append(fname)
    for obj in objs: cat_counter[obj.name] += 1
    sctool.save_voc_annotation(fname,
                               objs,  # bbox can be applied directly
                               rotation_degree=0, xmlpath=xml_file_path_dst)
    return image0, mapperAto0


def make_rotated(imageA, objs, imgfilepath, A, dA, mapperAto0):
    '''
       Tricky thing is: (suppose Ix means rotateImage(I,x) can get corrected-no-rotation image)
         mapping between I0 IB needs IA as a relay: I0->IA->IB
    '''

    imageB, mapperAtoB = sctool.rotateImage(imageA, dA)  # now image1 has rotation A-dA
    B = A - dA

    # a closuer
    def mapper0toB(pts0):
        ptsA = mapperAto0(pts0, src2dst=False)
        ptsB = mapperAtoB(ptsA, src2dst=True)
        return ptsB

    objs = copy.deepcopy(objs)

    # now rotate BBOXes accordingly
    for obj in objs:
        x0 = obj.bndbox.xmin
        y0 = obj.bndbox.ymin
        x1 = obj.bndbox.xmax
        y1 = obj.bndbox.ymax
        area_core = (y1 - y0) * (x1 - x0)

        if 0:
            pts0 = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).T
            ptsB = mapper0toB(pts0)
            pts = ptsB.T.reshape((-1, 1, 2))
            cv2.polylines(imageB, [pts], True, (0, 255, 255), 3)
            cv2.imshow("imageB", imageB);
            if ((cv2.waitKey(0) & 255) == ord('q')):
                sys.exit(0)

        pts = mapper0toB(np.array([[(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]).T).T

        # the bounding box's diagonal was used to encode A-dA
        obj.bndbox.xmin, obj.bndbox.ymin = np.amin(pts, 0)
        obj.bndbox.xmax, obj.bndbox.ymax = np.amax(pts, 0)

        w = obj.bndbox.xmax - obj.bndbox.xmin
        h = obj.bndbox.ymax - obj.bndbox.ymin

        if h * 8 < w:
            h = w / 8
            obj.bndbox.ymin -= h / 2
            obj.bndbox.ymax += h / 2
        elif w * 8 < h:
            w = h / 8
            obj.bndbox.xmin -= w / 2
            obj.bndbox.xmax += w / 2

        area_bbox = w * h
        # the box's name (class) was used to encode which diagonal is (A-dA)
        # when the tower is occupying less than 90% of the bbox
        if (area_core * 100 < area_bbox * 90 or (abs(B) > 45)):
            if (B > 6):
                obj.name += "L"

            if (B < -6):
                obj.name += "R"
        # leave all other as "c"

        # incorrcect rotation, just delete it
        if area_core > area_bbox:
            obj.name = None

    objs = [obj for obj in objs if obj.name is not None]

    if(len(objs) > 0):
        fname = filename_add_suffix(imgfilepath, "r{}".format(int(B)))
        cv2.imwrite(fname, imageB)
        list_fnames.append(fname)
        for obj in objs: cat_counter[obj.name] += 1
        sctool.save_voc_annotation(fname,
                                   objs,
                                   rotation_degree=0, xmlpath=xml_file_path_dst)



for ind, fimg in enumerate(img_file_list_src):

    print("{:.1%}".format(float(ind) / len(img_file_list_src)), fimg, end="\r")

    objs, A = sctool.load_voc_annotation(fimg, xmlpath=xml_file_path_src)

    image = cv2.imread(fimg)

    if len(objs) == 0 or (image is None):
        continue

    # corrected image
    image0, mapperAto0 = make_corrected(image, objs, fimg, A)

    make_rotated(image, objs, fimg, A, 0, mapperAto0)

    # now randomly select dA so B=A-dA is within (-90,90)
    # or dA is within (A-90,A+90) but since we rotate from A by dA, dA is also need to be in (-90,90)

    # target rotation angle
    B_list = np.concatenate([np.random.choice(181, 6)-90, np.random.choice(11, 12)-5])

    for B in B_list:
        dA = A - B
        if dA >= -90 and dA <= 90:
            make_rotated(image, objs, fimg, A, dA, mapperAto0)


# save list file
with open(os.path.join(list_file_path_dst, "train.txt"), "w") as f:
    for filepath in list_fnames:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        f.write(basename + "\n")
print("\n===========\n")
print(cat_counter)