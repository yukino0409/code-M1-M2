import glob
import cv2
import os
import numpy as np

#重ね合わせたマッピング結果を保存するディレクトリ
dir_save=r"E:\Liu\20240405\add_results\alpbha_20deg\\"
os.makedirs(dir_save,exist_ok=True)
dir_save_whole=r"E:\Liu\20240405\add_results\\"
os.makedirs(dir_save_whole,exist_ok=True)

imgs=glob.glob(r"E:\Liu\20240405\piecewise_regression\alpha_20deg\fill*.tif")
imgs_rotate=glob.glob(r"E:\Liu\20240405\piecewise_regression\alpha_20deg\rotate_90deg\fill\*.tif")

files=os.listdir(r"E:\Liu\20240405\piecewise_regression\alpha_20deg\fill\\")
numbers=[a[0:7] for a in files]

max_value=255
black=np.zeros_like(cv2.imread(imgs[0],0))
for i in range(len(imgs)):
    img=(cv2.imread(imgs[i],0)).astype(np.int32)
    img_rotate=(cv2.imread(imgs_rotate[i],0)).astype(np.int32)
    img_add=img+img_rotate
    img_add=(np.where(img_add>=max_value,max_value,0)).astype(np.uint8)

    """
    if (i==0)or(i==1): #積層欠陥は除外
        continue
    """
    black+=(img_add//max_value)

    name=numbers[i]+".tif"
    cv2.imwrite(dir_save+name,img_add)

black=(np.where(black>0,max_value,0)).astype(np.uint8)
cv2.imwrite(dir_save_whole+"whole_result_20deg.tif",black)
    

print("fin.")