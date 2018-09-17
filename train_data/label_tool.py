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

class VOC_Label_Window(object):
    def __init__(self, name, img_file_list, xml_file_path, img_scale = 1.0):
        self.name = name
        self.state = 0
        self.boxes = []
        self.curbox = sctool.bbox()
        self.ctrl_point = 1
        self.img_id = -1
        self.img_scale = img_scale
        self.img_scale_last = 0
        self.img_file_list = img_file_list
        self.xml_file_path = xml_file_path
        self.editor = None
        self.lastname = "noname"
        cv2.namedWindow(self.name)
        cv2.setMouseCallback(self.name, VOC_Label_Window._mouseCallback, self)
        self.imload(0)
        pass
    
   
    def imload(self, img_id):
    
        N = len(self.img_file_list)
        while(img_id < 0): img_id += N
        while(img_id >= N): img_id -=N
        
        do_reload = img_id == self.img_id
            
        # before switching, save & load 
        if self.img_id >= 0 and (not do_reload):
            objs = [sctool.build_voc_object(b.name, b.x0, b.y0, b.x1, b.y1) for b in self.boxes]
            sctool.save_voc_annotation(self.img_file_list[self.img_id], objs, xmlpath=self.xml_file_path)

        self.img_id = img_id
        self.bg0 = cv2.imread(self.img_file_list[self.img_id])
        self.img_scale_last = 0
        
        objs = sctool.load_voc_annotation(self.img_file_list[self.img_id], xmlpath=self.xml_file_path)
        self.boxes = [sctool.bbox.fromvoc(o) for o in objs]
        self.curbox = None
        self.imupdate()
    
    def ptmap(self, t):
        return ((int)(self.img_scale * t[0]),(int)(self.img_scale * t[1]))
        
    def ptmap_r(self, t):
        return ((int)(t[0]/self.img_scale),(int)(t[1]/self.img_scale))
    
    def _bbcolor(self, bb, reverse=False):
        class_id = hash(bb.name) & 7
        color = [0,0,0]
        if class_id & 1: color[0] = 255
        if class_id & 2: color[1] = 255
        if class_id & 4: color[2] = 255
        if reverse:
            color[0] = max(128,255 - color[0])
            color[1] = max(128,255 - color[1])
            color[2] = max(128,255 - color[2])
        return color
        
    def imupdate(self):
        if self.img_scale != self.img_scale_last:
            self.bg = cv2.resize(self.bg0, (0,0), fx=self.img_scale, fy=self.img_scale)
            self.img_scale_last = self.img_scale

        fg = np.copy(self.bg)
        
        if self.curbox:
            color = self._bbcolor(self.curbox, True)
            cv2.rectangle(fg, self.ptmap(self.curbox.tl), self.ptmap(self.curbox.br), color, thickness=6, lineType=8)

        for bb in self.boxes:
            color = self._bbcolor(bb, False)
            cv2.rectangle(fg, self.ptmap(bb.tl), self.ptmap(bb.br), color, thickness=2)
            cv2.line(fg,self.ptmap(bb.corner(1)), self.ptmap(bb.corner(3)), color, thickness=2)
            cv2.line(fg,self.ptmap(bb.corner(2)), self.ptmap(bb.corner(4)), color, thickness=2)
            cv2.putText(fg, bb.name, self.ptmap(bb.tl), cv2.FONT_HERSHEY_COMPLEX, 1, color,2)

        if self.curbox:
            # cv2.rectangle(fg, self.ptmap(self.curbox.tl), self.ptmap(self.curbox.br), (0,255,255), thickness=4, lineType=8)
            
            if self.ctrl_point >= 1 and self.ctrl_point <= 4:
                cv2.circle(fg, self.ptmap(self.curbox.corner(self.ctrl_point)), 15, (0,0,255), 1)
                
            if self.editor is not None:
                x0, y0 = self.ptmap(self.curbox.tl)
                x1, y1 = self.ptmap(self.curbox.br)
                fg[y0-40:y0+10, x0:x1, :] = np.array([0,0,0])
                cv2.putText(fg, "["+self.editor+"]", self.ptmap(self.curbox.tl), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255),2)
        
        title = "{}/{} {}".format(self.img_id, len(self.img_file_list), os.path.basename(self.img_file_list[self.img_id]))
        
        
        cv2.putText(fg, title, (0, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,100),2)
        cv2.imshow(self.name, fg)
    
    def mainloop(self):
        while(1):
            key = cv2.waitKey(10)
            if key == 27: break;
            if key > 0:
                self.keyDown(key)
    
    @staticmethod
    def _mouseCallback(event,x,y,flags,this):
        this.mouseCallback(event,x,y,flags)
    
    def mouseCallback(self,event,x,y,flags):
        
        x,y = self.ptmap_r((x,y))
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.curbox = sctool.bbox()
            self.curbox.name = self.lastname
            self.pt0 = (x, y)
            self.curbox.loc = self.pt0 + self.pt0
            self.state = 1
            pass
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.state == 1:
                # update drawing
                self.curbox.loc = self.pt0 + (x,y)
                self.imupdate()
            pass
        elif event == cv2.EVENT_LBUTTONUP:
            if self.state == 1:
                # mouse up, a point is ready
                self.curbox.loc = self.pt0 + (x,y)
            
                if self.curbox.area() < 9:
                    self.curbox = None
                    # mouse click, select one rect
                    bb_match_best = None
                    for bb in self.boxes:
                        if (x,y) in bb:
                            if bb_match_best:
                                if bb_match_best.area() > bb.area():
                                    bb_match_best = bb
                            else:
                                bb_match_best = bb
                    self.curbox = bb_match_best
                    self.imupdate()
                else:
                    self.boxes.append(self.curbox)
                    self.imupdate()
            
            self.state = 0
            pass

    def keyDown(self, key):
        if self.curbox:
            if key == ord('1'): self.ctrl_point = 1
            if key == ord('2'): self.ctrl_point = 2
            if key == ord('3'): self.ctrl_point = 3
            if key == ord('4'): self.ctrl_point = 4
            
            if key == ord('t'):
                # editor mode, user can input string by waitKey
                name = ''
                while(1):
                    self.editor = name
                    self.imupdate()
                    k = cv2.waitKey(0)
                    if k >= ord('a') and k <= ord('z'): name += chr(k)
                    if k == 27: break
                    if k == 8 and len(name) > 0: name = name[:-1]
                    if k == ord('\r'):
                        self.lastname = name
                        self.curbox.name = name
                        break

                self.editor = None
                self.imupdate()   
                
            if self.ctrl_point >= 1 and self.ctrl_point <= 4:
                if self.ctrl_point == 1 or self.ctrl_point == 2:
                    if key == ord('w'): self.curbox.y0 -= 1
                    if key == ord('s'): self.curbox.y0 += 1
                else:
                    if key == ord('w'): self.curbox.y1 -= 1
                    if key == ord('s'): self.curbox.y1 += 1
                    
                if self.ctrl_point == 1 or self.ctrl_point == 4:
                    if key == ord('a'): self.curbox.x0 -= 1
                    if key == ord('d'): self.curbox.x0 += 1
                else:
                    if key == ord('a'): self.curbox.x1 -= 1
                    if key == ord('d'): self.curbox.x1 += 1
            
            if key == 255:
                self.boxes.remove(self.curbox)
                self.curbox = None
            self.imupdate()
            
        if key == ord('=') or key == ord('+'):
            self.img_scale *= 1.1
            self.imupdate()
            
        if key == ord('-') or key == ord('_'):
            self.img_scale /= 1.1
            self.imupdate()
        
        if key == ord('r'):
            self.imload(self.img_id) # reload
            
        if key == 8:
            self.imload(self.img_id - 1)
            
        if key == 32:
            self.imload(self.img_id + 1)
        pass

helpstr='''
======================================
*                                    *
*   Welcome to use VOC label tools   *
*                                    *
*      by ltqusst @2018              *
======================================
Tips:

    use mouse to:
        drag a rectangle as the bounding box of new object
        click a rectangle to select it as current object
        
    use keyboard to:
        '1','2','3','4': select one corner of the rectangle as current corner
        'w','s':         move current corner up/down
        'a','d':         move current corner left/right
        '+','-':         zoom in/out current image
        (space):         move to next image     (after saving current annotations as XML)
        (backspace):     move to previous image (after saving current annotations as XML)
        'r':             reload current image and annotations, discard any changes
        't':             type the name for current object, you can use backspace delete character, return to confirm
'''

print(helpstr)

img_file_list = glob.glob("C:/LTQ/data/new_500/1*.jpg")
xml_file_path = "C:/LTQ/data/Annotations"

wnd = VOC_Label_Window("Label", img_file_list, xml_file_path, img_scale=2.5)
wnd.mainloop()


