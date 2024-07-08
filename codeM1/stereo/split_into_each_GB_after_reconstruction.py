import numpy as np
import os
import cv2
import math
import csv
import glob
#粒界のエッジに対してステレオ再構成した後に、単一粒界ごとに分割するプログラム
#分離後のzx平面の出力先のディレクトリ
dir_save=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\\"
os.makedirs(dir_save,exist_ok=True)
for i in range(3):
    os.makedirs(dir_save+"zx_"+str(i+1).zfill(3),exist_ok=True)
#粒界のエッジピクセルの三次元座標等が格納されたcsvファイルを読み込む
path=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\\"
with open(path+"y1_x1_z1_GBnum_GBside.csv") as f:
    reader=csv.reader(f)
    data=list(reader)
    data=np.array(data)
    coordinates=data.astype(float).astype(int) #[y_1,x_1,z_1,GB_num(1or2or3),side(1:left,2:right)]
z_min=np.min(coordinates[:,2]) #z_optの最小値が0になるようにオフセット
coordinates[:,2]-=z_min #[y_1,x_1,z_1_offset,GB_num,side]
z_shape,y_shape,x_shape=np.max(coordinates[:,2])+1,123,109 #以降pythonの3次元の軸に合わせてy,x,z→z,y,xに変更
#粒界のエッジピクセルの外側のみを抽出した画像のディレクトリ
outside_edges_deg1=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\matching_preparation\0deg\*.tif")

max_value=255

def after_np_where(array):
    array=(np.array(array)).flatten()
    array=array.reshape([-1,2],order='F')
    return array

for i in range(3): #単一粒界ごとに処理

    index_target=np.where(coordinates[:,3]==i+1)
    outside_edge=cv2.imread(outside_edges_deg1[i],0)
    index_outside_edge=after_np_where(np.where(outside_edge>=max_value//2)) #粒界のエッジピクセルの中でも外側のものだけを抽出する場合に使用
    black_3D=np.zeros((z_shape,y_shape,x_shape),dtype=np.uint8) #3次元空間はpythonの座標に従って[z,y,x]で指定 #csvファイルの出力時と順番を入れ替えるから少しややこしいかも

    for num_t in index_target[0]:
        y_1,x_1,z_1_offset,_,side=coordinates[num_t]

        #粒界のエッジピクセルの中でも外側のものだけを抽出する
        if (np.where((index_outside_edge[:,0]==y_1)&(index_outside_edge[:,1]==x_1))[0]).size!=1:
            continue
        
        if side==1: #粒界の左端
            black_3D[z_1_offset,y_1,x_1]=max_value
        else: #粒界の右端
            black_3D[z_1_offset,y_1,x_1]=max_value//2
    
    for j in range(y_shape): #zx平面を出力
        name=str(j).zfill(3)+".tif"
        cv2.imwrite(dir_save+"zx_"+str(i+1).zfill(3)+"\\"+name,black_3D[:,j,:])


print("fin.")