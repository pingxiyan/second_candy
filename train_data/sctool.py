#!/usr/bin/python
#encoding: utf-8
import os
import glob
import cv2
import numpy as np
import random
from time import time
import xml.etree.cElementTree as ET
from easydict import EasyDict as edict
import math

def build_voc_object(name, x0, y0, x1, y1):
    obj = edict()
    obj.name = name
    obj.bndbox = edict()
    obj.bndbox.xmin = x0
    obj.bndbox.ymin = y0
    obj.bndbox.xmax = x1
    obj.bndbox.ymax = y1
    return obj
    

class bbox(object):
    def __init__(self, other=None):
        if other is None:
            self.x0 = -1
            self.x1 = -1
            self.y0 = -1
            self.y1 = -1
            self.name = ""
        else:
            self.x0 = other.x0
            self.x1 = other.x1
            self.y0 = other.y0
            self.y1 = other.y1
            self.name = other.name

    @classmethod
    def fromvoc(cls, obj):
        ret = cls()
        ret.x0 = obj.bndbox.xmin
        ret.x1 = obj.bndbox.xmax
        ret.y0 = obj.bndbox.ymin
        ret.y1 = obj.bndbox.ymax
        ret.name = obj.name
        return ret
    
    def __contains__(self, m):
        return m[0] >= self.x0 and m[0] <= self.x1 and m[1] >= self.y0 and m[1] <= self.y1 
    
    def corner(self, i):
        if i == 1: return (self.x0, self.y0)
        if i == 2: return (self.x1, self.y0)
        if i == 3: return (self.x1, self.y1)
        if i == 4: return (self.x0, self.y1)
        return (self.x0, self.y0)

    @property
    def tl(self):
        return (self.x0, self.y0)
        
    @tl.setter
    def tl(self, value):
        self.x0, self.y0 = value
        
    @property
    def br(self):
        return (self.x1, self.y1)
        
    @br.setter
    def br(self, value):
        self.x1, self.y1 = value
    
    @property
    def loc(self):
        return (self.x0, self.y0, self.x1, self.y1)
        
    @loc.setter
    def loc(self, value):
        self.x0, self.y0, self.x1, self.y1 = value
        self.check_order()
    
    def check_order(self):
        if self.x0 > self.x1: self.x0, self.x1 = self.x1, self.x0
        if self.y0 > self.y1: self.y0, self.y1 = self.y1, self.y0
    
    def area(self):
        return max(0,  (self.y1 - self.y0)*(self.x1 - self.x0))
    
    def __and__(self, other):
        xA = max(self.x0, other.x0)
        yA = max(self.y0, other.y0)
        xB = min(self.x1, other.x1)
        yB = min(self.y1, other.y1)
        return max(0, xB - xA + 1) * max(0, yB - yA + 1)

    def __or__(self, other):
        xA = max(self.x0, other.x0)
        yA = max(self.y0, other.y0)
        xB = min(self.x1, other.x1)
        yB = min(self.y1, other.y1)
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        return float(self.area() + other.area() - interArea)

        
    def IOU(self, other):
        return (self & other)/(self | other)

    def make_square(self):
        ret = bbox(self)
        w = ret.x1 - ret.x0
        h = ret.y1 - ret.y0
        
        d = abs(w - h)
        d0 = int(d/2)
        d1 = d - d0
        
        if w > h:
            ret.y0 -= d0
            ret.y1 += d0
        else:
            ret.x0 -= d0
            ret.x1 += d1
        return ret


def get_rotMapper(height, width, angle):
    cosA = math.cos(angle * math.pi/180.0)
    sinA = math.sin(angle * math.pi/180.0)
    
    if (angle >= 0):
        dx = 0
        dy = width * sinA
    else:
        dx = -height * sinA
        dy = 0
    
    rotMat = np.array([[cosA,  sinA, dx],
                       [-sinA, cosA, dy]])
    
    rotBack = np.copy(rotMat)
    rotBack[1,0], rotBack[0,1] = rotBack[0,1], rotBack[1,0]
    rotBack[:,2] = np.matmul(rotBack[:2,:2],  -rotMat[:,2])
    
    def mapper(xy, src2dst = True, rounding = True):
        xy = np.array(xy).reshape((2,-1))
        if src2dst:
            ret = np.matmul(rotMat[:2,:2], xy) + rotMat[:,2].reshape((2,1))
        else:
            ret = np.matmul(rotBack[:2,:2], xy) + rotBack[:,2].reshape((2,1))
            
        if rounding:
            ret = ret.astype(np.int32)
        return ret
    return rotMat, rotBack, mapper


def rotateImage(image, angle):
    # this function will return exactly same image when angle is zero
    height, width = image.shape[:2]
    
    rotMat, rotBack, mapper = get_rotMapper(height, width, angle)
    
    bound_w, bound_h = np.maximum(np.maximum(mapper((width, height)), mapper((width, 0))), np.maximum(mapper((0, 0)), mapper((0, height))))
    
    result = cv2.warpAffine(image, rotMat, (bound_w, bound_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(110,0,30))
    return result, mapper

if 0:
    #debug rotateImage
    img = cv2.imread("/home/hddl/Pictures/Webcam/card/2018-08-27-121722.jpg")
    h,w = img.shape[:2]
    r,mapper = rotateImage(img, 20)
    
    print(r.shape, img.shape)
    #print("r == img ? {}".format((r == img).all()))
    
    pts = mapper(np.array([[0,0],[w,0],[w,h],[0,h]]).T).T.reshape((-1,1,2))
    print(pts)
    cv2.polylines(r, [pts] , True, (0,255,255),3)

    cv2.imshow("XXX", r)
    cv2.waitKey(0)

# {"name":xxx, "bndbox":{"xmin":0,"ymin":0,"xmax":0,"ymax":0}, ...}
# rotation_degree is the degree of rotaion needs to be made on image to get correct labeling
# the rotation is made so no raw image info is missing
def save_voc_annotation(image_filename, objs, rotation_degree = 0, img = None, folder="VOC2007", 
                        xmlpath="Annotations", common_info={"difficult":"0"}):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = image_filename
    ET.SubElement(annotation, "rotation_degree").text = str(rotation_degree)
    
    if img is None:
        newimg = cv2.imread(image_filename)
        W = newimg.shape[1]
        H = newimg.shape[0]
        depth = newimg.shape[2]
    else:
        W = img.shape[1]
        H = img.shape[0]
        depth = img.shape[2]
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(W)
    ET.SubElement(size, "height").text = str(H)
    ET.SubElement(size, "depth").text = str(depth)
    ET.SubElement(annotation, "segmented").text = str(0)
    
    for o in objs:
        obj = ET.SubElement(annotation, "object")
        
        for k, v in o.items():
            if isinstance(v, dict):
                subroot = ET.SubElement(obj, k)
                for k2, v2 in v.items():
                    ET.SubElement(subroot, str(k2)).text = str(v2)
            else:
                ET.SubElement(obj, str(k)).text = str(v)
                
        for k, v in common_info.items():
            ET.SubElement(obj, str(k)).text = str(v)

    tree = ET.ElementTree(annotation)
    
    filename = os.path.splitext(os.path.basename(image_filename))[0]
    xmlfilename = "{}/{}.xml".format(xmlpath, filename)
    
    tree.write(xmlfilename)


def load_voc_annotation(image_filename, xmlpath="Annotations"):

    filename = os.path.splitext(os.path.basename(image_filename))[0]
    xmlfilename = "{}/{}.xml".format(xmlpath, filename)
    objs = []
    rotation_degree = 0
    
    try:
        root = ET.parse(xmlfilename).getroot()
    except IOError:
        print("XML file {} doesn't exists!".format(xmlfilename))
        return objs, rotation_degree
    
    rot = root.findall('rotation_degree')
    if rot: 
        rotation_degree = float(rot[0].text)
    else:
        print("XML file {} has no rotaion!".format(xmlfilename))
    
    for o in root.findall('object'):
        
        name = o.find('name').text
        
        x0 = o.find('bndbox/xmin').text
        x1 = o.find('bndbox/xmax').text
        y0 = o.find('bndbox/ymin').text
        y1 = o.find('bndbox/ymax').text
        
        objs.append(build_voc_object(name, int(x0), int(y0), int(x1), int(y1)))
    
    return objs, rotation_degree

    
