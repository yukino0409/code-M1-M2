import numpy as np
import os
import cv2
import glob
import math

dir_save=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\\"
os.makedirs(dir_save,exist_ok=True)
dir_save_zx=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\zx\\"
os.makedirs(dir_save_zx,exist_ok=True)

imgs_deg1=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\matching_preparation\0deg\*.tif")
imgs_deg2=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\matching_preparation\20deg\*.tif")

max_value=255
h,w=(cv2.imread(imgs_deg1[0],0)).shape[:2]
deg1=0
deg2=20
tilt_rad=math.radians(deg2-deg1)
axis_x=45

#np.whereを用いてインデックスを取得する場合、この関数を通して2D配列に変換すると処理しやすい
def after_np_where(array):
    array=(np.array(array)).flatten()
    array=array.reshape([-1,2],order='F')
    return array


num=1
left_result=[]
right_result=[]
for i in range(len(imgs_deg1)):
    left_deg1=after_np_where(np.where((cv2.imread(imgs_deg1[i],0))==max_value))
    right_deg1=after_np_where(np.where((cv2.imread(imgs_deg1[i],0))==max_value//2))
    left_deg2=after_np_where(np.where((cv2.imread(imgs_deg2[i],0))==max_value))
    right_deg2=after_np_where(np.where((cv2.imread(imgs_deg2[i],0))==max_value//2))

    left_result_each=[]
    right_result_each=[]

    for y_l_1,x_l_1 in left_deg1:
        index_match_l=np.where(left_deg2[:,0]==y_l_1)
        if len(index_match_l[0])==0:
            continue
        else:
            y_l_2=(left_deg2[index_match_l[0],0])[0]
            x_l_2=(left_deg2[index_match_l[0],1])[0]
            left_result.append([y_l_1,x_l_1,y_l_2,x_l_2])
            left_result_each.append([y_l_1,x_l_1,y_l_2,x_l_2])
    
    for y_r_1,x_r_1 in right_deg1:
        index_match_r=np.where(right_deg2[:,0]==y_r_1)
        if len(index_match_r[0])==0:
            continue
        else:
            y_r_2=(right_deg2[index_match_r[0],0])[0]
            x_r_2=(right_deg2[index_match_r[0],1])[0]
            right_result.append([y_r_1,x_r_1,y_r_2,x_r_2])
            right_result_each.append([y_r_1,x_r_1,y_r_2,x_r_2])

np.savetxt(dir_save+"matching_left.csv",left_result,delimiter=",")
np.savetxt(dir_save+"matching_right.csv",right_result,delimiter=",")


#z座標算出
list_3d=[]

for y_l_1,x_l_1,y_l_2,x_l_2 in left_result:

    if y_l_1!=y_l_2: #保険
        break

    x_l_1_from_axis=x_l_1-axis_x
    x_l_2_from_axis=x_l_2-axis_x

    z_l_1=round((x_l_2_from_axis/math.sin(tilt_rad))-(x_l_1_from_axis/math.tan(tilt_rad)))

    list_3d.append([y_l_1,x_l_1,z_l_1]) #[y_1,x_1,z_1]

for y_r_1,x_r_1,y_r_2,x_r_2 in right_result:

    if y_r_1!=y_r_2: #保険
        break

    x_r_1_from_axis=x_r_1-axis_x
    x_r_2_from_axis=x_r_2-axis_x

    z_r_1=round((x_r_2_from_axis/math.sin(tilt_rad))-(x_r_1_from_axis/math.tan(tilt_rad)))

    list_3d.append([y_r_1,x_r_1,z_r_1]) #[y_1,x_1,z_1]
    
array_3d=np.array(list_3d)
np.savetxt(dir_save+"y1x1z1.csv",array_3d,delimiter=',')


#zをz_minでオフセットするとz軸の向きが反対になる(?)
z_min=np.min(array_3d[:,2])
thickness=np.max(array_3d[:,2])-z_min+1 #試料厚み#z=0を含むため+1する

#3次元配列に結果(座標)を反映
result_array_3d=np.zeros((h,w,thickness),dtype=np.uint8) #結果を格納する3次元配列(thicknessは上記の通り単純な引算)

for k in range(len(array_3d)):
    #z_coordinate_opt=(thickness-1)-(array_3d[k,2]-z_min) #z座標の向きに注意
    z_coordinate_opt=array_3d[k,2]-z_min #z_optの最小値が0になるようにオフセット
    result_array_3d[array_3d[k,0],array_3d[k,1],z_coordinate_opt]=max_value

for l in range(h):

    xz=result_array_3d[l,:,:] #[x,z]
    zx=np.transpose(xz,(1,0))
  
    name=str(l).zfill(3)+".tif"
    cv2.imwrite(dir_save_zx+name,zx)


print("fin.")