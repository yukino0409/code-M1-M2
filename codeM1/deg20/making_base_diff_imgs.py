import glob
import cv2
import os
import numpy as np
#各粒における平均化した回折図形を出力し、同時に粒番号を画素値とするナンバリング画像を作成する

#平均化した回折図形を保存するディレクトリ
dir_save=r"E:\Liu\20240405\MLS\base_diff_imgs\beta_20deg\\"
os.makedirs(dir_save,exist_ok=True)
dir_save_num=r"E:\Liu\20240405\mask_imgs\beta_20deg\whole\\"
os.makedirs(dir_save_num,exist_ok=True)

#回折図形(生データ,tif)のディレクトリ
imgs=glob.glob(r"E:\Liu\20240405\diff_img_raw\beta_20deg\selected\h76_w113\*.tif")

#ASTARの方位マッピングデータから作成したマスク画像のディレクトリ
masks=glob.glob(r"E:\Liu\20240405\mask_imgs\beta_20deg\*.tif")

#各種パラメータ設定
whole_area_h=76
whole_area_w=113
max_value=255

numbering=np.zeros((whole_area_h,whole_area_w),dtype=np.uint8)

for i in range(len(masks)):
    mask=cv2.imread(masks[i],0)
    index_mask=(np.array(np.where(mask==max_value))).flatten()
    index_mask=index_mask.reshape([-1,2],order='F') 

    list_img_stack=[]
    num_check=0
    for y,x in index_mask:
        file_num=y*whole_area_w+x
        img=cv2.imread(imgs[file_num],0)
        list_img_stack.append(img)

        #ナンバリング画像を作成
        numbering[y,x]=i+1

    array_img_stack=np.array(list_img_stack,dtype=np.float64)
    img_average=np.average(array_img_stack,axis=0)
    img_min=np.amin(img_average)
    img_average=img_average-img_min
    img_max=np.amax(img_average)
    img_average=(img_average/img_max)*max_value
    img_average=img_average.astype(np.uint8)
    cv2.imwrite(dir_save+str(i+1).zfill(3)+".tif",img_average)

cv2.imwrite(dir_save_num+"numbering.tif",numbering)
    


print(1)
