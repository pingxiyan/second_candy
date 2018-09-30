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

img_file_list_dst = "/home/hddl/dockerv0/data/voc/VOCdevkit/VOC2020/JPEGImages"
xml_file_path_dst = "/home/hddl/dockerv0/data/voc/VOCdevkit/VOC2020/Annotations"
list_file_path_dst = "/home/hddl/dockerv0/data/voc/VOCdevkit/VOC2020/ImageSets/Main"


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


def make_rotated(imageA, objs, imgfilepath, A, dA, mapperAto0, Bmin=0):
    '''
       Tricky thing is: (suppose Ix means rotateImage(I,x) can get corrected-no-rotation image)
         mapping between I0 IB needs IA as a relay: I0->IA->IB
    '''
    imageB, mapperAtoB = sctool.rotateImage(imageA, dA)  # now image1 has rotation A-dA
    B = A - dA

    # a closuer (input is Nx2 points, output is bbox
    def get_bbox_mapper0toB(pts0):
        ptsA = mapperAto0(pts0.T, src2dst=False)
        ptsB = mapperAtoB(ptsA, src2dst=True)

        nx0, ny0 = np.amin(ptsB, 1)
        nx1, ny1 = np.amax(ptsB, 1)
        w = nx1 - nx0
        h = ny1 - ny0

        nx0 = max(nx0, 0)
        ny0 = max(ny0, 0)
        nx1 = min(nx1, imageB.shape[1])
        ny1 = min(ny1, imageB.shape[0])
        return nx0, ny0, nx1, ny1, w, h, w*h

    objs = copy.deepcopy(objs)

    # now rotate BBOXes accordingly
    for obj in objs:
        x0 = obj.bndbox.xmin
        y0 = obj.bndbox.ymin
        x1 = obj.bndbox.xmax
        y1 = obj.bndbox.ymax
        area_core = (y1 - y0) * (x1 - x0)

        # the bounding box's diagonal was used to encode A-dA
        nx0, ny0, nx1, ny1, w, h, area_bbox = get_bbox_mapper0toB(np.array([[(x0 + x1) / 2, y0], [(x0 + x1) / 2, y1]]))

        # the box's name (class) was used to encode which diagonal is (A-dA)
        # when the tower is occupying less than 90% of the bbox
        type_suffix = ""
        #if (area_core * 100 < area_bbox * 90 or (abs(B) > 45)):
        if (B > Bmin):
            type_suffix = "L"
        if (B < -Bmin):
            type_suffix = "R"

        # leave all other as "c"
        obj.name += type_suffix
        if not type_suffix:
            # if we decide to keep the type as C, just make type of bbox
            nx0, ny0, nx1, ny1, w, h, area_bbox = get_bbox_mapper0toB(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))

        obj.bndbox.xmin = nx0
        obj.bndbox.xmax = nx1
        obj.bndbox.ymin = ny0
        obj.bndbox.ymax = ny1

    fname = filename_add_suffix(imgfilepath, "r{}".format(int(B)))
    cv2.imwrite(fname, imageB)
    list_fnames.append(fname)
    for obj in objs: cat_counter[obj.name] += 1
    sctool.save_voc_annotation(fname,
                               objs,
                               rotation_degree=0, xmlpath=xml_file_path_dst)


def load_sample(fimg, xmlpath, bHflip):
    objs, A = sctool.load_voc_annotation(fimg, xmlpath=xmlpath)
    image = cv2.imread(fimg)
    if len(objs) == 0 or (image is None):
        return objs, A, image

    if bHflip:
        image0, mapperAto0 = sctool.rotateImage(image, A)
        image = image[:,::-1,:]
        W = image0.shape[1]
        for obj in objs:
            obj.bndbox.xmin, obj.bndbox.xmax = W - obj.bndbox.xmax, W - obj.bndbox.xmin
        A = -A
    return objs, A, image


for HorizontalFlip in [False, True]:
    for ind, imgfilepath in enumerate(img_file_list_src):

        fimg = imgfilepath
        if HorizontalFlip:
            fimg = filename_add_suffix(fimg, "H")

        print("{:.1%}".format(float(ind) / len(img_file_list_src)), fimg, end="\r")

        objs, A, image = load_sample(imgfilepath, xml_file_path_src, HorizontalFlip)
        if len(objs) == 0 or (image is None):
            continue

        # corrected image
        image0, mapperAto0 = make_corrected(image, objs, fimg, A)

        make_rotated(image, objs, fimg, A, 0, mapperAto0)

        # now randomly select dA so B=A-dA is within (-90,90)
        # or dA is within (A-90,A+90) but since we rotate from A by dA, dA is also need to be in (-90,90)

        # target rotation angle

        def get_BList(angle_settings):
            all_angle = []
            for rg in angle_settings:
                all_angle.append(np.random.choice(rg[1] - rg[0] + 1, rg[2]) + rg[0])
            B_list = np.concatenate(all_angle)
            return B_list

        N = 2
        # generate "cL/cR" with perturbation
        B_list0 = get_BList([(3,88,N),(-88,-3,N)])
        for B in B_list0:
            dA = A - B
            if dA >= -90 and dA <= 90:
                make_rotated(image, objs, fimg, A, dA, mapperAto0, Bmin = 2)

        # generate "c" with perturbation
        B_list1 = get_BList([(1, 4, 1), (-4, 1, 1)])
        for B in B_list1:
            dA = A - B
            if dA >= -90 and dA <= 90:
                # Bmin set to 90 to disable cL/cR generation
                make_rotated(image, objs, fimg, A, dA, mapperAto0, Bmin=90)

# save list file
with open(os.path.join(list_file_path_dst, "train.txt"), "w") as f:
    for filepath in list_fnames:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        f.write(basename + "\n")
print("\n===========\n")
print(cat_counter)