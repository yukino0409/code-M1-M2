import numpy as np
import os
import cv2
import glob
import math

#傾斜軸がxy平面内で回転していたため、補正してマッチングおよび再構成を行うプログラム

dir_save=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\\"
os.makedirs(dir_save,exist_ok=True)
dir_save_matching=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\matching_line\\" #ピクセルマッチングの方向および範囲を出力
os.makedirs(dir_save_matching,exist_ok=True)
os.makedirs(dir_save_matching+"left",exist_ok=True)
os.makedirs(dir_save_matching+"right",exist_ok=True)
dir_save_zx=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\zx\\"
os.makedirs(dir_save_zx,exist_ok=True)
dir_save_vis=r"E:\Liu\20240226_stereo_with_sato\stereo\matching_and_reconstruction\rotate_tilt-axis\zx\vis\\" #3次元可視化ソフト(visualizer)の軸とそろえた場合の出力結果
os.makedirs(dir_save_vis,exist_ok=True)

#粒界に対してエッジ抽出(左端:255,右端:255//2)および位置合わせした画像のディレクトリを指定
imgs_deg1=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\alignment\0deg\*.tif")
imgs_deg2=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\alignment\20deg\*.tif")

max_value=255
h,w=(cv2.imread(imgs_deg1[0],0)).shape[:2]
tilt_rad=math.radians(20) #傾斜角(tilt angle)
rad_tilt_axis=math.radians(79.5) #傾斜軸のxy平面内の回転角
rad_normal=rad_tilt_axis+np.pi/2 #傾斜軸に対する法線方向の角度
point_align_y=66 #アライメントした点の座標(アライメント後:YX座標系)
point_align_x=45
b_intercept=point_align_y-(point_align_x*np.tan(rad_tilt_axis)) #傾いた傾斜軸の切片(画像左上を原点とする座標系yx(デフォルト)) #傾いた傾斜軸に合わせて回転させた座標系をY'X'座標系と表記する
#ax+by+c=0に式変形すると係数は
a_coef=np.tan(rad_tilt_axis)
b_coef=-1
c_coef=b_intercept

#np.whereを用いてインデックスを取得する場合、この関数を通して2D配列に変換すると処理しやすい
def after_np_where(array):
    array=(np.array(array)).flatten()
    array=array.reshape([-1,2],order='F')
    return array

#点と直線の距離(非負値),回転させた座標系Y'X'におけるX'の符号を求める関数
def Calc_distance(point_x,point_y): #直線ax+by+c=0との距離を計算するだけなのでyx座標系の点(x0,y0)を代入する
    numer=abs(a_coef*point_x+b_coef*point_y+c_coef) #分子
    denom=math.sqrt(pow(a_coef,2)+pow(b_coef,2)) #分母
    #distanceの符号を決定する時は、実際に座標系を回転させるのではなく、反対回りに点を回転させる
    #point_x_rotated=((point_x-point_align_x)*math.cos((np.pi/2)-rad_tilt_axis))-((point_y-point_align_y)*math.sin((np.pi/2)-rad_tilt_axis)) 
    #point_y_rotated=point_x*math.sin(rad_tilt_axis)+point_y*math.cos(rad_tilt_axis)
    distance=numer/denom
    y_on_line=(np.tan(rad_tilt_axis)*point_x)+b_intercept   
    if point_y>y_on_line:
        distance=distance*(-1)
    return distance #Y'X'(回転座標系)におけるX'座標

num_left=0
num_right=0
left_result=[]
right_result=[]

for i in range(len(imgs_deg1)):
    left_deg1=after_np_where(np.where((cv2.imread(imgs_deg1[i],0))==max_value))
    right_deg1=after_np_where(np.where((cv2.imread(imgs_deg1[i],0))==max_value//2))

    for y_l_1,x_l_1 in left_deg1:
        if (y_l_1==96)&(x_l_1==74):
            print("check")
        black_left=np.zeros_like(cv2.imread(imgs_deg1[0],0))

        b=y_l_1-(math.tan(rad_normal)*x_l_1)
        #直線の終点の座標を求める（ただし、境界条件0<=x<w）
        y_end=round(math.tan(rad_normal)*(w-1)+b) #y=ax+b
        y_end_2=round(b) #y=a*0+b

        #粒界のエッジピクセルを始点にして両方向に線を絵画
        cv2.line(black_left,pt1=(x_l_1,y_l_1),pt2=(w-1,y_end),color=1,thickness=1,lineType=cv2.LINE_4) #pt1,pt2は(x,y)で指定
        cv2.line(black_left,pt1=(x_l_1,y_l_1),pt2=(0,y_end_2),color=1,thickness=1,lineType=cv2.LINE_4)

        img_deg2_left=np.where(cv2.imread(imgs_deg2[i],0)==max_value,1,0)
        img_deg2_left+=black_left
        name_left=str(num_left).zfill(3)+".tif"
        cv2.imwrite(dir_save_matching+"left\\"+name_left,img_deg2_left)
        index_matched_left=after_np_where(np.where(img_deg2_left==2))
        if len(index_matched_left)==0:
            continue
        
        #直線上に複数のピクセルがヒットした場合
        if len(index_matched_left)>1:
            list_angle=[]
            for y_sample,x_sample in index_matched_left:
                arctan=np.arctan((y_sample-y_l_1)/(x_sample-x_l_1))
                if arctan<0:
                    arctan+=np.pi
                sample_angle=abs(arctan-rad_normal)
                list_angle.append(sample_angle)          
            index_matched_left=(index_matched_left[list_angle.index(min(list_angle))]).reshape((1,2))
        y_l_2=index_matched_left[0,0]
        x_l_2=index_matched_left[0,1]
        left_result.append([y_l_1,x_l_1,y_l_2,x_l_2,i+1,1]) #単一粒界ごとに分けられるように、粒界の番号も格納しておく #GBの左右端も記録しておく(1:left,2:right)
        num_left+=1   


    for y_r_1,x_r_1 in right_deg1:
        black_right=np.zeros_like(cv2.imread(imgs_deg1[0],0))

        b=y_r_1-(math.tan(rad_normal)*x_r_1)
        #直線の終点の座標を求める（ただし、境界条件0<=x<w）
        y_end=round(math.tan(rad_normal)*(w-1)+b) #y=ax+b
        y_end_2=round(b) #y=a*0+b

        #粒界のエッジピクセルを始点にして両方向に線を絵画
        cv2.line(black_right,pt1=(x_r_1,y_r_1),pt2=(w-1,y_end),color=1,thickness=1,lineType=cv2.LINE_4) #pt1,pt2は(x,y)で指定
        cv2.line(black_right,pt1=(x_r_1,y_r_1),pt2=(0,y_end_2),color=1,thickness=1,lineType=cv2.LINE_4)

        img_deg2_right=np.where(cv2.imread(imgs_deg2[i],0)==max_value//2,1,0)
        img_deg2_right+=black_right
        name_right=str(num_right).zfill(3)+".tif"
        cv2.imwrite(dir_save_matching+"right\\"+name_right,img_deg2_right)
        index_matched_right=after_np_where(np.where(img_deg2_right==2))
        if len(index_matched_right)==0:
            continue
        
        #直線上に複数のピクセルがヒットした場合
        if len(index_matched_right)>1:
            list_angle=[]
            for y_sample,x_sample in index_matched_right:
                arctan=np.arctan((y_sample-y_r_1)/(x_sample-x_r_1))
                if arctan<0:
                    arctan+=np.pi
                sample_angle=abs(arctan-rad_normal)
                list_angle.append(sample_angle)          
            index_matched_right=(index_matched_right[list_angle.index(min(list_angle))]).reshape((1,2))
        y_r_2=index_matched_right[0,0]
        x_r_2=index_matched_right[0,1]
        right_result.append([y_r_1,x_r_1,y_r_2,x_r_2,i+1,2]) #単一粒界ごとに分けられるように、粒界の番号も格納しておく#GBの左右端も記録しておく(1:left,2:right)
        num_right+=1
    

np.savetxt(dir_save+"matching_left.csv",left_result,delimiter=",")
np.savetxt(dir_save+"matching_right.csv",right_result,delimiter=",")


#z座標算出
list_3d=[]
for y_l_1,x_l_1,y_l_2,x_l_2,num_GB_l,side_l in left_result:

    x_l_1_from_axis=Calc_distance(x_l_1,y_l_1)
    x_l_2_from_axis=Calc_distance(x_l_2,y_l_2)

    z_l_1=round((x_l_2_from_axis/math.sin(tilt_rad))-(x_l_1_from_axis/math.tan(tilt_rad)))

    list_3d.append([y_l_1,x_l_1,z_l_1,num_GB_l,side_l]) #[y_1,x_1,z_1,GB_num,side]

for y_r_1,x_r_1,y_r_2,x_r_2,num_GB_r,side_r in right_result:
    """
    #確認
    y_r_1,x_r_1,y_r_2,x_r_2=7,33,6,38
    """
    x_r_1_from_axis=Calc_distance(x_r_1,y_r_1)
    x_r_2_from_axis=Calc_distance(x_r_2,y_r_2)

    z_r_1=round((x_r_2_from_axis/math.sin(tilt_rad))-(x_r_1_from_axis/math.tan(tilt_rad)))

    list_3d.append([y_r_1,x_r_1,z_r_1,num_GB_r,side_r]) #[y_1,x_1,z_1,GB_num,side]
    
array_3d=np.array(list_3d)
np.savetxt(dir_save+"y1_x1_z1_GBnum_GBside.csv",array_3d,delimiter=',')


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
    zx_for_vis=cv2.flip(zx, 1)
  
    name=str(l).zfill(3)+".tif"
    cv2.imwrite(dir_save_zx+name,zx)
    cv2.imwrite(dir_save_vis+name,zx_for_vis)



print("fin.")