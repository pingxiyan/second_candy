#!/usr/bin/python
#encoding: utf-8
import os
import pygame
import cv2
import numpy as np
import random
from time import time
import xml.etree.cElementTree as ET
import sys
import math


class CamCalibrator(object):
    def __init__(self):
        pass
    
    # 3D point r
    def _map(self, r):
        k1,k2,p1,p2,k3 = self.dist
        fx = self.mtx[0,0]
        fy = self.mtx[1,1]
        cx = self.mtx[0,2]
        cy = self.mtx[1,2]

        Rrf = np.matmul(self.R, r.flatten())
        xyz = Rrf + self.t.flatten()
        x1 = xyz[0]/xyz[2]
        y1 = xyz[1]/xyz[2]
        r2 = x1*x1 + y1*y1
        dr = 1.0 + (k1+(k2+k3*r2)*r2)*r2
        x2 = x1*dr + 2*p1*x1*y1 + p2*(r2+2*x1**2)
        y2 = y1*dr + p1*(r2+2*y1**2) + 2*p2*x1*y1
        u = fx*x2+cx
        v = fy*y2+cy
        
        return u,v
    
    def map(self, x, y):
        return self.mapxy[int(y),int(x),0], self.mapxy[int(y),int(x),1]
    
    def gen_map(self, w, h):
        print("Generating map {}x{} ...".format(w,h))
        self.mapxy = np.zeros((h,w,2), np.float32)
        for y in range(h):
            for x in range(w):
                u,v = self._map(np.array([x,y,0]))
                self.mapxy[y,x,0] = u
                self.mapxy[y,x,1] = v
        print("Done")
        
    def start(self, cap, win_name, BG_W = 960, BG_H = 720):
        ChessBoardImage = np.ones((BG_H, BG_W, 3))

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((9*9,3), np.float32)

        ioxy = 0
        cy = 0
        for y in range(0, 10):
            cy = 1-cy
            cx = cy
            for x in range(0, 10):
                y0 = (y)*BG_H/10
                y1 = (y+1)*BG_H/10
                x0 = (x)*BG_W/10
                x1 = (x+1)*BG_W/10
                ChessBoardImage[y0:y1, x0:x1] = cx
                cx = 1-cx
                
                if(x+1 < 10 and y+1 < 10):
                    objp[ioxy,0] = x1
                    objp[ioxy,1] = y1
                    ioxy += 1        

        cv2.imshow(win_name,ChessBoardImage)

        print("Press any key when the target window is proper aligned to camera")
        
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame',frame)
            key = cv2.waitKey(40)
            if key != -1: break
                
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (9,9),None)

        objpoints = []
        imgpoints = []
        # If found, add object points, image points (after refining them)
        if ret == False:
            print("findChessboardCorners failed")
            sys.exit(0)
            
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)

        #print(type(corners), corners.shape, corners)
        #print(type(objp), objp.shape, objp)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (9,9), corners,ret)
        cv2.imshow('frame',frame)
        cv2.waitKey(0)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
        if ret == False:
            print("calibrateCamera failed")
            sys.exit(0)

        print("mtx({})={}\n\t dist({})={}\n\t rvecs({})={}\n\t tvecs({})={}".format(mtx.shape, mtx, dist[0].shape, dist[0], rvecs[0].shape, rvecs[0], tvecs[0].shape, tvecs[0]))

        # 3x3 Rotaion
        self.R = cv2.Rodrigues(rvecs[0])[0]
        self.t = tvecs[0]
        self.dist = dist[0]
        self.mtx = mtx
        self.gen_map(BG_W, BG_H)
        
        print("R({})={}".format(self.R.shape, self.R))
        
        for r in objp:
            u,v = self.map(r[0], r[1])
            cv2.circle(frame, (int(u), int(v)), 10, (55,255,155),2)
        
        cv2.imshow('frame2',frame)
        cv2.waitKey(0)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cc = CamCalibrator()
    cc.start(cap, 960,720)

    
