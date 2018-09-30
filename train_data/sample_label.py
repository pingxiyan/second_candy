#!/usr/bin/python
#encoding: utf-8
import os
import pygame
import cv2
import numpy as np
import random
from time import time
import xml.etree.cElementTree as ET
import calib

head_rng = range(int(0xb0), int(0xf7) + 1)
body_rng = range(int(0xa1), int(0xfe) + 1)
words = []
for head in head_rng:
    for body in body_rng:
        words.append('{:x}{:x}'.format(head, body).decode('hex').decode('gb2312', 'ignore'))

words =  u'一二三四五六七八九十个百千万大小多少上下左右里外开关出入方圆远近长短前后来去日月水火山石田土天地星云风雨雪电花草树木红黄蓝绿江河湖海春夏秋冬车船枪炮人口手足头耳眼牙坐立走跑男女老幼爷奶哥姐弟妹门窗桌椅米面饭菜瓜果桃李猪牛羊马鸟兽鱼虫虎象'

words =  u'春夏秋冬'

pygame.init()

BG_W = 960
BG_H = 960*3/4

font = pygame.font.Font("KaiTi.ttf", 256)

with open("bglist.txt", "r") as f:
    bglist = f.readlines()

def generate_voc_annotation(image_filename, img, objs, folder="VOC2007"):
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = image_filename
    
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img.shape[1])
    ET.SubElement(size, "height").text = str(img.shape[1])
    ET.SubElement(size, "depth").text = str(3)
    
    ET.SubElement(annotation, "segmented").text = str(0)
    
    for (name, xmin, ymin, xmax, ymax) in objs:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "obj"
        ET.SubElement(obj, "hanzi").text = name
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
    
    tree = ET.ElementTree(annotation)
    tree.write("Annotations/{}.xml".format(os.path.splitext(image_filename)[0]))


def generate_next_train_sample(bglist, words, font):
    
    # randome BG
    fname = bglist[random.choice(range(0,len(bglist)))].rstrip('\n')
    bg_img = cv2.imread(fname)
    bg_img = cv2.resize(bg_img, (BG_W,BG_H))
        
    #font size
    F = random.choice(range(60, bg_img.shape[0]/2))
    
    # append text (space separated)
    cnt = (bg_img.shape[1]-1)/(F+30)
    txt = [words[random.choice(range(0,len(words)))] for _ in range(cnt)]
    
    # total margin (above + below)
    mL = random.choice(range(0, min(2*F, bg_img.shape[0] - F - 1 - 1)))
    my = random.choice(range(0, bg_img.shape[0] - F - 1 - 1 - mL))
    
    # where text is put
    y = my + random.choice(range(0, mL))
    
    # add banner
    mean_bg = np.mean(bg_img[my:my+mL+F,:,:]) 
    if mean_bg < 180:
        bg_img[my:my+mL+F,:,:] = np.array((255,255,255))
    
    print("mean_bg=", mean_bg)
    
    loc = []
    x = random.choice(range(0, max(1,bg_img.shape[1] - len(txt)*(F+30))))
    for t in txt:
        timg = font.render(t, True, (0, 0, 0), (255, 255, 255))
        timg = pygame.surfarray.array3d(timg)
        timg = timg.transpose([1,0,2])
        timg = cv2.resize(timg, (F,F), interpolation=cv2.INTER_AREA)
        
        H = timg.shape[0]
        W = timg.shape[1]
        
        # draw text on bg_img
        #bg_img[y:y+H, x:x+W] = timg
        for dy in range(timg.shape[0]):
            for dx in range(timg.shape[1]):
                if timg[dy,dx,0] < 200:
                    bg_img[y+dy, x+dx, :] = timg[dy,dx]
        
        loc.append([t.encode("utf-8").encode("hex"), x, y, x+W, y+H])
        
        x += W + 30
    print("loc=",loc)
    return bg_img, loc




def calibrate(cap):
    sample_image = np.ones((BG_H, BG_W, 3))
    MAP_W = 10
    MAP_H = MAP_W * BG_H / BG_W
    marker_r = 20
    
    cmd = 0
    
    # give user some time to setup window location
    for my in range(0, MAP_H+1):
        y = my * (BG_H-1) / (MAP_H)
        sample_image[y,:] = 0
        
    for mx in range(0, MAP_W+1):
        x = mx * (BG_W-1) / (MAP_W)
        sample_image[:,x] = 0
    
    cv2.imshow('sample_image',sample_image)
    while(cmd != ord(' ')):
        ret, frame = cap.read()
        cv2.imshow('frame',frame)
        cmd = cv2.waitKey(10) & 0xFF

    sample_image = sample_image*0 + 1
    
    # calibration
    locMap = np.zeros((MAP_H+1, MAP_W+1, 2), dtype=np.int)

    for my in range(0, MAP_H+1):
        for mx in range(0, MAP_W+1):
            
            y = my * BG_H / (MAP_H)
            x = mx * BG_W / (MAP_W)
            
            image = np.copy(sample_image)
            image[max(0,y-marker_r):min(BG_H,y+marker_r), max(0,x-marker_r):min(BG_W,x+marker_r), 0] = 0
            image[max(0,y-marker_r):min(BG_H,y+marker_r), max(0,x-marker_r):min(BG_W,x+marker_r), 1] = 0
            image = cv2.GaussianBlur(image, (marker_r*2+1,marker_r*2+1), 0)
            
            cv2.imshow('sample_image',image)
            
            for i in range(10):
                ret, frame = cap.read()
                
                # do calib (show the detected mark)
                frame_blur = cv2.blur(frame, (15,15)).astype(np.float)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(frame_blur[:,:,0] + frame_blur[:,:,1] - frame_blur[:,:,2])
                
                locMap[my, mx, 0] = min_loc[0]
                locMap[my, mx, 1] = min_loc[1]
                
                cv2.circle(frame, min_loc, 5, (55,255,155),2)
                cv2.imshow('frame',frame)
                
                cmd = cv2.waitKey(10) & 0xFF
                if cmd == ord('q'): break

    locMap2 = cv2.resize(locMap.astype(np.float), (BG_W, BG_H), interpolation=cv2.INTER_LINEAR)
    locMap2 = np.round(locMap2).astype(np.int)
    print(locMap2[0,0,:])
    print(locMap[0,0,:])
    print(locMap2[-1,-1,:])
    print(locMap[-1,-1,:])  
        
    return locMap2
    
            
cap = cv2.VideoCapture(0)
#locMap = calibrate(cap)

cc = calib.CamCalibrator()
cc.start(cap, "sample_image", 960,720)

id = 0
while(True):
    # generate next train sample
    sample_image, loc = generate_next_train_sample(bglist, words, font)
    id += 1
    
    cv2.imshow('sample_image',sample_image)

    for i in range(10):
        ret, frame = cap.read()
        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'): break

    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    backup = np.copy(frame)
    # generate the mapped loc
    for (_,x0,y0,x1,y1) in loc:
        
        rx0 = min(cc.map(x0,y0)[0],  cc.map(x0,y1)[0])
        rx1 = max(cc.map(x1,y0)[0],  cc.map(x1,y1)[0])
        ry0 = min(cc.map(x0,y0)[1],  cc.map(x1,y0)[1])
        ry1 = max(cc.map(x0,y1)[1],  cc.map(x1,y1)[1])
        cv2.rectangle(frame,(rx0,ry0),(rx1, ry1),(55,255,155),2)
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    key = cv2.waitKey(200) & 0xFF
    if key == ord('q'): break
    
    filename = str(id).zfill(10)+".jpg"
    cv2.imwrite(os.path.join("JPEGImages",filename), backup)
    generate_voc_annotation(filename, backup, loc)

    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


