import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import argparse, sys
import torch
#from ultralytics import YOLO
import open3d as o3d

def histo(img_gray:np.uint8):
    val = np.zeros(256, dtype=np.int32)
    height ,width = img_gray.shape
    print(height, width)
    for h in range(height):
        for w in range(width):
            val[img_gray[h,w]] +=1
    return val

def bias_seg(img_gray:np.uint8, bias:int=30):
    height, width = img_gray.shape
    img_seg = np.zeros((height, width), np.uint8)
    print(f"h:{height}, w:{width} img: {img_seg.shape}")
    for h in range(height):
        for w in range(width):
            if img_gray[h,w] > bias:
                img_seg[h,w] = 1
    return img_seg

def main(args=None):
    #model = YOLO("yolo26n-seg.pt")
    img = cv.imread(args.img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, np.array([50, 5, 5]), np.array([120, 150, 180]))
    #res = model(img) # "https://ultralytics.com/images/bus.jpg"
    img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img_hist = histo(img_gray)
    h_his = histo(hsv[:,:,0])
    #iseg = bias_seg(img_gray=hsv[:,:,0], bias=115)
    #plt.figure(1)
    #plt.imshow(img)
    plt.figure(1)
    plt.imshow(hsv)
    plt.figure(2)
    plt.imshow(mask)
    #plt.figure(2)
    #plt.title("h")
    #plt.plot(h_his)
    #plt.figure(5)
    #plt.plot(h_his)
    #plt.figure(6)
    #plt.imshow(mask)
    plt.show()
    #res[0].show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True)
    args = parser.parse_args()
    main(args=args)