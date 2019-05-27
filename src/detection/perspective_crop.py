import h5py
import scipy.io as io
import PIL.Image as Image
import os
import sys
import glob
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn import linear_model
from scipy.ndimage.filters import gaussian_filter 
import scipy
import cv2
import json
for video_id in range(1,101):
	with open("detection_results/test_framebyframe/video%d.txt"%(video_id),'r') as f:
	    dt_results_fbf = json.load(f)
	for frame in dt_results_fbf:
		if dt_results_fbf[frame]>12:
			imgs = dt_results_fbf[frame]
			break

	im = cv2.imread("data/AIC_Track3/processed_images/%d/1.jpg"%video_id))
	y = []
	h = []
	for box in imgs:
	    left_up, right_bottom = (int(box[0]), int(box[1])), (int(box[2]),int(box[3]))
	    cv2.rectangle(im, left_up, right_bottom, (0, 0, 255), 3)
	    y.append((box[1]+box[3])/2)
	    h.append(np.sqrt((box[3]-box[1])*(box[2]-box[0])))
	regr = linear_model.LinearRegression()
	regr.fit(np.array(y).reshape(-1, 1), np.array(h))
	a, b = regr.coef_, regr.intercept_


	tmp_results =[]

	if b>10:
	    start_h = b
	    start = 0
	else:
	    start_h = 10
	    start = (10-b)/a
	end = im.shape[0]-1
	width = im.shape[1]-1
	all_count = (1/a)*np.log(a*end+b)-(1/a)*np.log(start_h)
	stride = 3

	num = int(np.ceil(all_count/stride))
	stride2 = all_count/num
	points=[int(start)]
	tmp_start = (1/a)*np.log(start_h)

	crop_boxes =[]
	for i in range(1,num+1):
	    point = (np.exp((tmp_start+stride2)*a)-b)/a
	    if i ==num:
	        points[-1] = int((np.exp((tmp_start-stride+stride2)*a)-b)/a)
	        now_point = int((np.exp((tmp_start+stride)*a)-b)/a)
	    else:
	        now_point = int(point)
	        
	    wid_num = np.ceil(width/float((now_point-points[i-1])*2))
	    tmp_width = int((now_point-points[i-1])*2)
	    tmp_width_point=0
	    for j in range(1,int(wid_num)+1):
	        if j == wid_num:
	            tmp_box = [points[i-1],width-1-tmp_width,now_point,width-1]
	        else:
	            tmp_box = [points[i-1],tmp_width_point,now_point,tmp_width_point+tmp_width]

	        tmp = img_name_.replace('/','_')
	        # save_name = 'data/perspective_images/%s_%d_%d.jpg'%(tmp,i,j)
	        crop_boxes.append(tmp_box)
	        # cv2.imwrite(save_name,frame)
	        tmp_width_point +=  tmp_width
	    points.append(int(point))
	    tmp_start = tmp_start + stride2
	with open(str(video_id)+'.json','w') as json_file:
    	json.dump(crop_boxes,json_file)
