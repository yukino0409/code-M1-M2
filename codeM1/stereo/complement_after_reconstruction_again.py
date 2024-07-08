import numpy as np
import os
import cv2
import math
import glob
#ステレオ再構成した後の単一粒界(エッジの外側のみ抽出したもの)ごとに分離した3Dデータに対して、粒界の平面を埋めて可視化するプログラム
#補完後のzx平面の出力先のディレクトリ
dir_save=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\complement\\"
os.makedirs(dir_save,exist_ok=True)
for i in range(3):
    os.makedirs(dir_save+"zx_"+str(i+1).zfill(3),exist_ok=True)

path=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\\"

z_shape,y_shape,x_shape=40,123,109 #以降pythonの3次元の軸に合わせてy,x,z→z,y,xに変更
max_value=255

def after_np_where(array):
    array=(np.array(array)).flatten()
    array=array.reshape([-1,2],order='F')
    return array

for i in range(3): #単一粒界ごとに処理
    imgs=glob.glob(path+"zx_"+str(i+1).zfill(3)+"\*.tif")

    #zx平面を3次元配列に格納
    black_3D=np.zeros((z_shape,y_shape,x_shape),dtype=np.uint8)
    black_3D_res=np.zeros_like(black_3D)
    for j in range(len(imgs)):
        zx=cv2.imread(imgs[j],0)
        black_3D[:,j,:]=zx
    
    #y=nのzx平面上に2点、y=n+1のzx平面上に2点の計4点確認される場合を最初に（平面に）補完していく
    black_3D_bin=np.where(black_3D>=max_value//2,1,0) #点数の確認のため0or1で2値化した配列を作成しておく
    for j in range(y_shape-1):
        if np.sum(black_3D_bin[:,j,:])+np.sum(black_3D_bin[:,j+1,:])==4: #4点確認される場合

            #y=nのzx平面
            zx_n=black_3D[:,j,:] #2値化していない3次元配列からzx平面を抜き出す
            z_n_l,x_n_l=np.where(zx_n==max_value)[0][0],np.where(zx_n==max_value)[1][0]
            z_n_r,x_n_r=np.where(zx_n==max_value//2)[0][0],np.where(zx_n==max_value//2)[1][0]
            zx_n=cv2.line(zx_n,pt1=(x_n_l,z_n_l),pt2=(x_n_r,z_n_r),color=1,thickness=1,lineType=cv2.LINE_8)
            #y=y+1のzx平面
            zx_np1=(black_3D[:,j+1,:]).copy()
            z_np1_l,x_np1_l=np.where(zx_np1==max_value)[0][0],np.where(zx_np1==max_value)[1][0]
            z_np1_r,x_np1_r=np.where(zx_np1==max_value//2)[0][0],np.where(zx_np1==max_value//2)[1][0]
            zx_np1=cv2.line(zx_np1,pt1=(x_np1_l,z_np1_l),pt2=(x_np1_r,z_np1_r),color=1,thickness=1,lineType=cv2.LINE_8)

            zx_n+=zx_np1
            zx_n=cv2.line(zx_n,pt1=(x_n_l,z_n_l),pt2=(x_np1_l,z_np1_l),color=1,thickness=1,lineType=cv2.LINE_8)
            zx_n=cv2.line(zx_n,pt1=(x_n_r,z_n_r),pt2=(x_np1_r,z_np1_r),color=1,thickness=1,lineType=cv2.LINE_8)
            zx_n=(np.where(zx_n>0,max_value,0)).astype(np.uint8)
            contours,_=cv2.findContours(zx_n,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            zx_n=cv2.drawContours(zx_n,contours,-1,max_value,-1)
            #cv2.imwrite(dir_save+"test.tif",zx_n)
            black_3D_res[:,j,:]=zx_n
        
        elif np.sum(black_3D_bin[:,j,:])+np.sum(black_3D_bin[:,j+1,:])==3: #3点確認される場合
            index_min=np.argmin([np.sum(black_3D_bin[:,j,:]),np.sum(black_3D_bin[:,j+1,:])]) #1点の方のインデックスから左右端を取得
            val_single=np.max(black_3D[:,j+index_min,:])

            if index_min==0:
                zx_single=(black_3D[:,j,:]).copy()
                zx_double=black_3D[:,j+1,:]
            else:
                zx_single=(black_3D[:,j+1,:]).copy()
                zx_double=black_3D[:,j,:]
            
            z_single,x_single=np.where(zx_single==val_single)[0][0],np.where(zx_single==val_single)[1][0]
            z_double_l,x_double_l=np.where(zx_double==max_value)[0][0],np.where(zx_double==max_value)[1][0]
            z_double_r,x_double_r=np.where(zx_double==max_value//2)[0][0],np.where(zx_double==max_value//2)[1][0]

            zx_single=cv2.line(zx_single,pt1=(x_double_l,z_double_l),pt2=(x_double_r,z_double_r),color=1,thickness=1,lineType=cv2.LINE_8)
            zx_single=cv2.line(zx_single,pt1=(x_single,z_single),pt2=(x_double_l,z_double_l),color=1,thickness=1,lineType=cv2.LINE_8)
            zx_single=cv2.line(zx_single,pt1=(x_single,z_single),pt2=(x_double_r,z_double_r),color=1,thickness=1,lineType=cv2.LINE_8)

            zx_single=(np.where(zx_single>0,max_value,0)).astype(np.uint8)
            contours,_=cv2.findContours(zx_single,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            zx_single=cv2.drawContours(zx_single,contours,-1,max_value,-1)
            black_3D_res[:,j,:]=zx_single


            #singleで検出した側に繰り返し処理
            for k in range(1,y_shape,1):
                if index_min==0:
                    zx_single_next=(black_3D[:,j-k,:]).copy()
                else:
                    zx_single_next=(black_3D[:,j+1+k,:]).copy()
                
                if np.max(zx_single_next)==0:
                    break

                z_single_next,x_single_next=np.where(zx_single_next==val_single)[0][0],np.where(zx_single_next==val_single)[1][0]
                zx_single_next=cv2.line(zx_single_next,pt1=(x_single_next,z_single_next),pt2=(x_single,z_single),color=1,thickness=1,lineType=cv2.LINE_8)
                if val_single==max_value:
                    zx_single_next=cv2.line(zx_single_next,pt1=(x_single_next,z_single_next),pt2=(x_double_r,z_double_r),color=1,thickness=1,lineType=cv2.LINE_8)
                    zx_single_next=cv2.line(zx_single_next,pt1=(x_single,z_single),pt2=(x_double_r,z_double_r),color=1,thickness=1,lineType=cv2.LINE_8)
                else:
                    zx_single_next=cv2.line(zx_single_next,pt1=(x_single_next,z_single_next),pt2=(x_double_l,z_double_l),color=1,thickness=1,lineType=cv2.LINE_8)
                    zx_single_next=cv2.line(zx_single_next,pt1=(x_single,z_single),pt2=(x_double_l,z_double_l),color=1,thickness=1,lineType=cv2.LINE_8)
                
                zx_single_next=(np.where(zx_single_next>0,max_value,0)).astype(np.uint8)
                contours,_=cv2.findContours(zx_single_next,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                zx_single_next=cv2.drawContours(zx_single_next,contours,-1,max_value,-1)

                if index_min==0:
                    black_3D_res[:,j-k,:]=zx_single_next
                else:
                    black_3D_res[:,j+k,:]=zx_single_next
                
                x_single,z_single=x_single_next,z_single_next


    for j in range(y_shape):
        name=str(j).zfill(3)+".tif"
        cv2.imwrite(dir_save+"zx_"+str(i+1).zfill(3)+"\\"+name,black_3D_res[:,j,:])



print("fin.")