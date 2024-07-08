import numpy as np
import os
import cv2
import glob

#粒界のエッジ抽出およびアライメント後の画像に対して、外側の1ピクセルのみを抽出するプログラム

dir_save=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_preparation\\"
os.makedirs(dir_save,exist_ok=True)

imgs_0deg=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\alignment\0deg\*.tif")
imgs_20deg=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\alignment\20deg\*.tif")

max_value=255
h,w=(cv2.imread(imgs_0deg[0],0)).shape[:2]

#np.whereを用いてインデックスを取得する場合、この関数を通して2D配列に変換すると処理しやすい
def after_np_where(array):
    array=(np.array(array)).flatten()
    array=array.reshape([-1,2],order='F')
    return array

list_data=[] #0deg(3枚)→20deg(3枚)の順で格納
for i in range(len(imgs_0deg)):
    list_data.append(cv2.imread(imgs_0deg[i],0))
for i in range(len(imgs_20deg)):
    list_data.append(cv2.imread(imgs_20deg[i],0))

num=1
for img in list_data:

    y_start=min(np.where(img>=max_value//2)[0])
    y_end=max(np.where(img>=max_value//2)[0])

    black=np.zeros_like(img)
    for y in range(y_start,y_end+1,1):
        if len(np.where(img[y]==max_value)[0])>0:
            x_left=min(np.where(img[y]==max_value)[0])
            black[y,x_left]=max_value
        if len(np.where(img[y]==max_value//2)[0])>0:
            x_right=max(np.where(img[y]==max_value//2)[0])
            black[y,x_right]=max_value//2

    if num<=len(imgs_0deg):
        name=str(num).zfill(3)+".tif"
        os.makedirs(dir_save+"0deg",exist_ok=True)
        cv2.imwrite(dir_save+"0deg\\"+name,black)
    else:
        name=str(num-len(imgs_0deg)).zfill(3)+".tif"
        os.makedirs(dir_save+"20deg",exist_ok=True)
        cv2.imwrite(dir_save+"20deg\\"+name,black)
        
    
    num+=1
        


print("fin.")