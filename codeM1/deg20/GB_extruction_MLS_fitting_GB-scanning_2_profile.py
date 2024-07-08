import numpy as np
import os
import cv2
import glob
import csv
import statistics
import piecewise_regression
import matplotlib.pyplot as plt

#MLS結果に対して粒界を検出するコード(さらに、全ての折れ線回帰によるフィッティング結果のラインプロファイルを出力する)

dir_save=r"E:\Liu\20240405\piecewise_regression\alpha_20deg\\"
os.makedirs(dir_save,exist_ok=True)
dir_save_each=dir_save+"each_results\\"
os.makedirs(dir_save_each,exist_ok=True)
dir_save_fill=dir_save+"fill\\"
os.makedirs(dir_save_fill,exist_ok=True)
dir_save_mask=dir_save+"mask\\"
os.makedirs(dir_save_mask,exist_ok=True)

#ナンバリング画像を読み込む
numbering=cv2.imread(r"E:\Liu\20240405\MLS\numbering_map\alpha_20deg\numbering.tif",0)
#ASTARの方位マッピングデータから作成したマスク画像のディレクトリ
masks=glob.glob(r"E:\Liu\20240405\mask_imgs\alpha_20deg\*.tif")
#隣り合う粒のコンビネーションを出力したcsvファイルを読み込む
with open(r"E:\Liu\20240405\MLS\grain_combination_alpha_20deg.csv") as f:
    reader=csv.reader(f)
    data=list(reader)
    data=np.array(data)
    combination=data.astype(float).astype(int)
#MLS結果(マッピング)のディレクトリ
path=r"E:\Liu\20240405\MLS\mapping_GB-numbering\alpha_20deg\\"
#ASTARの粒界ピクセルを番号付けして分離した画像を読み込む
path_GB_numberings=r"E:\Liu\20240405\MLS\numbering_map\alpha_20deg\GB_numbering\\"

#各種パラメータ設定
max_value=255
fitting_range=25
n_breakpoints=2
h,w=numbering.shape[:2]

#np.whereを用いてインデックスを取得する場合、この関数を通して2D配列に変換すると処理しやすい
def after_np_where(array):
    array=(np.array(array)).flatten()
    array=array.reshape([-1,2],order='F') 
    return array


#x方向に連なっているピクセルを1ピクセル(代表点)に
index_GB=np.where(numbering==0)
index_GB=after_np_where(index_GB)
array_GB_center=np.zeros_like(numbering)
list_series=[]
for i in range(len(index_GB)):
    target=index_GB[i]

    #全粒界ピクセルの最後のピクセルの場合
    if i==len(index_GB)-1:
        previous=index_GB[i-1]
        #連なった粒界ピクセルの最後のピクセルの場合
        if target[1]-1==previous[1]:
            list_series.append(target[1])
            med=len(list_series)//2
            center=list_series[med]
        #1ピクセルの場合
        else:
            center=target[1]       
        array_GB_center[target[0],center]=1
        break

    next=index_GB[i+1]

    #連なった粒界ピクセルの途中のピクセルの場合
    if (target[0]==next[0])&(target[1]+1==next[1]):
        list_series.append(target[1])
        continue
    #連なった粒界ピクセルの最後のピクセルの場合
    elif len(list_series)!=0:
        list_series.append(target[1])
        med=len(list_series)//2
        center=list_series[med]
        list_series=[]
    #1ピクセルの場合
    else:
        center=target[1]
    array_GB_center[target[0],center]=1

cv2.imwrite(dir_save+"GB_center.tif",array_GB_center)


#ASTARのマッピング結果から生成した全てのマスク画像をリストに格納
list_masks=[]
for i in range(len(masks)):
    mask=cv2.imread(masks[i],0)
    list_masks.append(mask)


list_bp=[] #break pointの位置を格納する
for num_base_1,num_base_2 in combination:

    dir_save_each_num=dir_save_each+str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+"\\"
    os.makedirs(dir_save_each_num,exist_ok=True)
    dir_save_each_num_csv=dir_save_each_num+"csv\\"
    os.makedirs(dir_save_each_num_csv,exist_ok=True)
    dir_save_each_num_bp=dir_save_each_num+"bp\\"
    os.makedirs(dir_save_each_num_bp,exist_ok=True)

    '''
    #確認用#要コメントアウト
    num_base_1=8
    num_base_2=9
    '''

    list_num_base=[num_base_1,num_base_2]
    mask_base1=list_masks[num_base_1-1]
    mask_base2=list_masks[num_base_2-1]
    mask_base1=mask_base1//max_value
    mask_base2=mask_base2//max_value

    path_GB_numbering=path_GB_numberings+str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+".tif"
    GB_numbering=cv2.imread(path_GB_numbering,0)

    GB_matched=(GB_numbering//max_value)+array_GB_center
    index_GB_each=np.where(GB_matched==2)
    index_GB_each=after_np_where(index_GB_each)

    GB_matched_2=np.where(GB_matched==2,1,0)
    mask=mask_base1+mask_base2+(GB_numbering//max_value)
    mask=(np.where(mask>=1,1,0)).astype(np.uint8)
    name_mask=str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+".tif"
    cv2.imwrite(dir_save_mask+name_mask,mask)


    #粒界ピクセルの左側の粒番号を取得する
    list_left_num=[]
    for y_c,x_c in index_GB_each:

        for j in range(1,w,1):
            if x_c-j<0:
                break
            else:
                left_num=numbering[y_c,x_c-j]
                if left_num==0:
                    continue
                elif (left_num==num_base_1)or(left_num==num_base_2):
                    list_left_num.append(left_num)
                    break
                else:
                    break

    left_num=statistics.mode(list_left_num)
    #左側が白であるMLS結果を読み込む
    dir_MLS=path+str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+"\\"+str(left_num).zfill(3)+".tif"
    img_MLS=cv2.imread(dir_MLS,0)

    
    #複数直線によるフィッティング
    result_array_each=np.zeros_like(numbering)
    result_array_each_fill=np.zeros_like(numbering)
    for y_c,x_c in index_GB_each:

        """
        #確認用#要コメントアウト
        y_c,x_c=49,24
        """

        #フィッティングする視野を選択
        #計算対象とする領域が画像の枠からはみ出ている場合
        if (x_c-fitting_range<0)or(x_c+fitting_range+1>w):
            black=np.zeros(2*fitting_range+1,dtype=np.int64)
            black_mask=np.zeros(2*fitting_range+1,dtype=np.int64)
            if x_c-fitting_range<0:
                MLS_target=(img_MLS[y_c,0:x_c+fitting_range+1]).astype(np.int64) #フィッティングする視野を選択
                mask_target=(mask[y_c,0:x_c+fitting_range+1]).astype(np.int64)
                cutting_width=MLS_target.shape[0]
                black[2*fitting_range+1-cutting_width:]=MLS_target
                black_mask[2*fitting_range+1-cutting_width:]=mask_target
            else:
                MLS_target=(img_MLS[y_c,x_c-fitting_range:w]).astype(np.int64) #フィッティングする視野を選択
                mask_target=(mask[y_c,x_c-fitting_range:w]).astype(np.int64)
                cutting_width=MLS_target.shape[0]
                black[:cutting_width]=MLS_target
                black_mask[:cutting_width]=mask_target
            MLS_target=black
            mask_target=black_mask
        else:
            MLS_target=(img_MLS[y_c,x_c-fitting_range:x_c+fitting_range+1]).astype(np.int64)
            mask_target=(mask[y_c,x_c-fitting_range:x_c+fitting_range+1]).astype(np.int64)
        
        mask_target_zero=np.where(mask_target==0)[0] #マスクの白ピクセル≡MLS計算領域 #返り値は1次元配列
        len_zeros=len(mask_target_zero)
        #ターゲット配列にマスク外(黒)が含まれている場合
        if len_zeros!=0:
            #粒界ピクセル(自動的にtarget配列の中央ピクセル)からマスク外に当たるまで左スキャン
            for scan in range(fitting_range+1):
                x_scan=fitting_range-scan
                if mask_target[x_scan]==0:
                    left_value=MLS_target[x_scan+1]
                    MLS_target[0:x_scan+1]=left_value
                    break
                else:
                    continue
            #粒界ピクセル(自動的にtarget配列の中央ピクセル)からマスク外に当たるまで右スキャン
            for scan in range(fitting_range+1):
                x_scan=fitting_range+scan
                if mask_target[x_scan]==0:
                    right_value=MLS_target[x_scan-1]
                    MLS_target[x_scan:2*fitting_range+1]=right_value
                    break
                else:
                    continue

        xx=np.arange(x_c-fitting_range,x_c+fitting_range+1,1)
        """
        #確認用
        xx=np.arange(0,2*fitting_range+1,1)
        """
        #fitting
        pw_fit=piecewise_regression.Fit(xx,MLS_target,n_breakpoints=n_breakpoints)

        pw_results=pw_fit.get_results()
        x_bp1=round(((pw_results["estimates"])["breakpoint1"])["estimate"])
        x_bp2=round(((pw_results["estimates"])["breakpoint2"])["estimate"])
        
        if x_bp1<0:
            x_bp1=0
        elif x_bp1>=w:
            x_bp1=w-1
        if x_bp2<0:
            x_bp2=0
        elif x_bp2>=w:
            x_bp2=w-1
        
        list_bp.append([y_c,x_bp1,x_bp2])
        result_array_each[y_c,x_bp1]=max_value
        #result_array_each[y_c,x_bp2]=max_value
        result_array_each[y_c,x_bp2]=max_value//2 #粒界の一端は255,他端は127で出力しておくと再構成時に楽かも(?)
        result_array_each_fill[y_c,x_bp1:x_bp2+1]=max_value

        
        #確認用#要コメントアウト

        pw_fit.plot_data(color="b")
        pw_fit.plot_fit(color="r")
        #pw_fit.summary()
        pw_fit.plot_breakpoints(color="green")
        #test=pw_fit.bootstrap_data(xx,MLS_target)
        #plt.subplots(figsize=(8,5),tight_layout=True)
        #fig, ax = plt.subplots(figsize=(8,5),tight_layout=True)
        #xticklabels = ax.get_xticklabels()
        #ax.set_xticklabels(xticklabels,fontsize=18, rotation=45)
        #ax.set_xlabel("x",fontsize=22)
        name_fit=str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+"_"+str(y_c).zfill(3)+"_"+str(x_c).zfill(3)+".png"
        name_profile=str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+"_"+str(y_c).zfill(3)+"_"+str(x_c).zfill(3)+"_profile.csv"
        name_bp=str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+"_"+str(y_c).zfill(3)+"_"+str(x_c).zfill(3)+"_bp.csv"
        plt.grid()    
        plt.plot(x_bp1,MLS_target[np.where(xx==x_bp1)[0][0]],marker='.',color='g',markersize=12)
        plt.plot(x_bp2,MLS_target[np.where(xx==x_bp2)[0][0]],marker='.',color='g',markersize=12)
        #plt.show()
        plt.savefig(dir_save_each_num+name_fit)
        MLS_target_csv=(np.hstack((xx,MLS_target))).reshape([fitting_range*2+1,2],order="F")
        np.savetxt(dir_save_each_num_csv+name_profile,MLS_target_csv,delimiter=",")
        list_for_profile=list([y_c,x_bp1,x_bp2])
        np.savetxt(dir_save_each_num_bp+name_bp,list_for_profile,delimiter=",")
        plt.close()
        
    name=str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+"_ref"+str(left_num).zfill(3)+".tif"
    cv2.imwrite(dir_save_each+name,result_array_each)
    cv2.imwrite(dir_save_fill+name,result_array_each_fill)


#全体の粒界マッピング結果を出力する
result_array=np.zeros_like(numbering)
result_array_2=np.zeros_like(numbering)
result_array_bin=np.zeros_like(numbering)
for y,x_bp1,x_bp2 in list_bp:
    result_array[y,x_bp1]=max_value
    result_array[y,x_bp2]=max_value//2
    result_array_2[y,x_bp1]=max_value
    result_array_2[y,x_bp2]=max_value
    result_array_bin[y,x_bp1:x_bp2+1]=max_value
cv2.imwrite(dir_save+"result.tif",result_array)
cv2.imwrite(dir_save+"result_both_max-val.tif",result_array_2)
cv2.imwrite(dir_save+"result_fill.tif",result_array_bin)


print("fin.")