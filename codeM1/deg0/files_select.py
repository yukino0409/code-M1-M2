import numpy as np
import cv2
import glob
import os
#粒界抽出する領域のみの回折図形を出力するコード（二度手間ではあるため、diff_imgs_extruction_from_blockfile.pyとまとめてしまっても良い）

dir_save=r"E:\Liu\20240405\diff_img_raw\alpha_0deg\selected\h82_w115\\"
os.makedirs(dir_save,exist_ok=True)

imgs=glob.glob(r"E:\Liu\20240405\diff_img_raw\alpha_0deg\*.tif") #生の回折図形のディレクトリを指定して、関数globを使用してすべての回折図形のディレクトリを取得しておく
path=r"E:\Liu\20240405\diff_img_raw\alpha_0deg\\" #上と同じディレクトリを指定
files=os.listdir(path) #回折図形の元々のファイル名を取得しておく（ファイル名を変更しない）

w_defo=290 #回折図形を取得した全体の視野のwを入力

y_offset=88 #粒界抽出領域のy座標のスタート座標を指定する
h=82 #粒界抽出領域のy方向の幅（高さ）を指定する
x_offset=160 #xも上記と同様に指定する
w=115

for y in range(y_offset,y_offset+h,1):
    for x in range(x_offset,x_offset+w,1):
        
        num=y*w_defo+x #任意の(y,x)座標からimgsの何番目に目的の回折図形が格納されているかを取得する
        img=cv2.imread(imgs[num],0) #回折図形を画像で読み込む（第2引数を0で指定するとグレースケール（uint8）で読み込む）

        cv2.imwrite(dir_save+files[num],img) #そのまま回折図形を画像として出力する



print("fin.")