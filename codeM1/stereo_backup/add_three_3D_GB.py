import numpy as np
import os
import cv2
import math
import csv
import glob

#粒界のエッジに対してステレオ再構成した後に、粒界の平面を埋めて可視化するプログラム
dir_save=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\complement\add_result\\"
os.makedirs(dir_save,exist_ok=True)
dir_save_vis=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\complement\add_result\for_vis\\"
os.makedirs(dir_save_vis,exist_ok=True)

imgs_1=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\complement\zx_001\*.tif")
imgs_2=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\complement\zx_002\*.tif")
imgs_3=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\complement\zx_003\*.tif")

max_value=255
list_data=[]
#black_3D=np.zeros((40,123,109),dtype=np.uint8) #(z,y,x)
for i in range(len(imgs_1)):
    img_1=(cv2.imread(imgs_1[i],0)).astype(np.int32)
    img_2=(cv2.imread(imgs_2[i],0)).astype(np.int32)
    img_3=(cv2.imread(imgs_3[i],0)).astype(np.int32)
    img_stack=img_1+img_2+img_3
    img_stack=(np.where(img_stack>=max_value,max_value,0)).astype(np.uint8)
    img_stack_for_vis=cv2.flip(img_stack, 1)
    #black_3D[:,i,:]=img_stack
    name=str(i).zfill(3)+".tif"
    cv2.imwrite(dir_save+name,img_stack)
    cv2.imwrite(dir_save_vis+name,img_stack_for_vis)


print("fin.")




