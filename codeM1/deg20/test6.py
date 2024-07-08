import glob
import cv2
import os
import numpy as np
import csv
import statistics

#マッピング結果を保存するディレクトリ
dir_save=r"E:\Liu\20240405\MLS\mapping_GB-numbering\beta_20deg\\"
os.makedirs(dir_save,exist_ok=True)
#ナンバリングした画像を保存するディレクトリ
dir_save_num=r"E:\Liu\20240405\MLS\numbering_map\beta_20deg\\"
os.makedirs(dir_save_num,exist_ok=True)

#回折図形(生データ,tif)のディレクトリ
imgs=glob.glob(r"E:\Liu\20240405\diff_img_raw\beta_20deg\selected\h76_w113\*.tif")
#ASTARの方位マッピングデータから作成したマスク画像のディレクトリ
masks=glob.glob(r"E:\Liu\20240405\mask_imgs\beta_20deg\*.tif")
#MLSの基底となる平均化した回折図形のディレクトリ
bases=glob.glob(r"E:\Liu\20240405\MLS\base_diff_imgs\beta_20deg\*.tif")
#隣り合う粒のコンビネーションを出力したcsvファイルを読み込む
with open(r"E:\Liu\20240405\MLS\grain_combination_beta_20deg.csv") as f:
    reader=csv.reader(f)
    data=list(reader)
    data=np.array(data)
    combination=data.astype(float).astype(int)

#各種パラメータ設定
VBF_h=76
VBF_w=113
max_value=255
kernel_size=5

"""
#透過スポットを除外する場合
DB_y_offset=133
DB_x_offset=137
DB_hw=17
list_index_DB=[]
for DB_y in range(DB_y_offset,DB_y_offset+DB_hw,1):
    for DB_x in range(DB_x_offset,DB_x_offset+DB_hw,1):
        index_DB_1D=DB_y*VBF_w+DB_x
        list_index_DB.append(index_DB_1D)
"""
#np.whereを用いてインデックスを取得する場合、この関数を通して2D配列に変換すると処理しやすい
def after_np_where(array):
    array=(np.array(array)).flatten()
    array=array.reshape([-1,2],order='F') 
    return array

#uint8の場合のコントラスト最適化
def contrast_opt(array,max_value):
    color_min=np.amin(array) #毎回(粒ごとに)最小値を計算しなおす必要がある
    array=array-color_min #img_minの正負にかかわらず最小値を0にする
    color_max=np.amax(array)
    array=(array/color_max)*max_value
    array=array.astype(np.uint8)
    return array

# ASTARで出力された粒界ピクセルに対する番号付け
def numbering_GB(array, kernel_size):  # ナンバリング画像をインプット
    index = np.where(array == 0)
    index = after_np_where(index)
    half_kernel_size = kernel_size // 2
    list_results = []
    
    for y, x in index:
        if (y - half_kernel_size < 0) & (x - half_kernel_size < 0):
            array_target = (array[0:kernel_size + (y - half_kernel_size), 0:kernel_size + (x - half_kernel_size)]).flatten()
        elif y - half_kernel_size < 0:
            array_target = (array[0:kernel_size + (y - half_kernel_size), x - half_kernel_size:x + half_kernel_size + 1]).flatten()
        elif x - half_kernel_size < 0:
            array_target = (array[y - half_kernel_size:y + half_kernel_size + 1, 0:kernel_size + (x - half_kernel_size)]).flatten()
        else:
            array_target = (array[y - half_kernel_size:y + half_kernel_size + 1, x - half_kernel_size:x + half_kernel_size + 1]).flatten()
        
        array_target = array_target[~(array_target == 0)]  # e.g.)array_target=[2,2,2,0,0,2,2] ->array_target=[2,2,2,2,2] |array_target=[2,2,2,0,0,2,2,1,1]->array_target=[2,2,2,2,2,1,1]
        print(y, x)
        multimode = statistics.multimode(array_target)  # 複数の最頻値が出る場合に対応した関数（リストで返される） e.g.)multimode=2 |multimode=2
        
        if len(multimode) == 1:
            array_target = array_target[~(array_target == multimode[0])]  # e.g.)array_target=[2,2,2,2,2]からmultimode=2以外を抽出->array_target=[] (empty)|array_target=[2,2,2,2,2,1,1]からmultimode=2以外を抽出->array_target=[1,1]
            
            if len(array_target) == 0:
                print(f"Warning: Empty array_target after removing mode at coordinates ({y}, {x})")
                continue  # 空の配列の場合は次のループに進む
                
            mode_second = statistics.mode(array_target)  # 2番目の最頻値を求める場合は1つだけ最頻値を抽出する e.g.)emptyの配列からmodeを求めようとしている|array_target=[1,1]->mode [1]
            multimode.append(mode_second)
        
        # 昇順に変更してリストに格納
        if multimode[0] > multimode[1]:
            list_results.append([y, x, multimode[1], multimode[0]])
        else:
            list_results.append([y, x, multimode[0], multimode[1]])
    
    array_results = np.array(list_results)
    return array_results



#マスク画像を使って各ピクセルに粒のナンバリングした画像を作成
numbering=np.zeros((VBF_h,VBF_w),dtype=np.uint8)
list_masks=[]
for i in range(len(masks)):
    mask=cv2.imread(masks[i],0)
    list_masks.append(mask)
    index_mask=np.where(mask==max_value)
    index_mask=after_np_where(index_mask)
    num_check=0
    for y,x in index_mask:
        numbering[y,x]=i+1 
cv2.imwrite(dir_save_num+"numbering.tif",numbering)


#粒界ピクセルにもナンバリング
array_numbering_GB=numbering_GB(numbering,kernel_size)
os.makedirs(dir_save_num+"GB_numbering",exist_ok=True)
os.makedirs(dir_save_num+"GB_numbering\\num_opt",exist_ok=True)
list_GB_numbering=[]
num=0
for num_a,num_b in combination:
    black=np.zeros_like(numbering)
    index=np.where((array_numbering_GB[:,2]==num_a)&(array_numbering_GB[:,3]==num_b))[0]
    for i in index:
        y=array_numbering_GB[i,0]
        x=array_numbering_GB[i,1]
        black[y,x]=max_value
    name=str(num_a).zfill(3)+"_"+str(num_b).zfill(3)+".tif"
    name_2=str(num).zfill(3)+".tif"
    cv2.imwrite(dir_save_num+"GB_numbering\\"+name,black)
    cv2.imwrite(dir_save_num+"GB_numbering\\num_opt\\"+name_2,black)
    list_GB_numbering.append(black)
    num+=1


#MLS
os.makedirs(dir_save+"cal_area",exist_ok=True)
num_for_GB_pixels=0
for num_base_1,num_base_2 in combination:
    list_num_base=[num_base_1,num_base_2]
    dir_save_map=dir_save+str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+"\\"
    os.makedirs(dir_save_map,exist_ok=True)

    #計算(MLS)対象となるピクセルの座標を格納しておく
    mask_base1=list_masks[num_base_1-1]
    mask_base2=list_masks[num_base_2-1]
    GB_numbering=list_GB_numbering[num_for_GB_pixels]
    mask_base1=mask_base1//max_value
    mask_base2=mask_base2//max_value
    GB_numbering=GB_numbering//max_value
    cal_area=(mask_base1+mask_base2+GB_numbering).astype(np.uint8)
    name=str(num_base_1).zfill(3)+"_"+str(num_base_2).zfill(3)+".tif"
    cv2.imwrite(dir_save+"cal_area\\"+name,cal_area)

    index_mask_all=np.where(cal_area>=1)
    index_array=after_np_where(index_mask_all)


    #2次元配列の作成
    original=[]
    for y,x in index_array:
        num_file=y*VBF_w+x
        img=cv2.imread(imgs[num_file],0)
        img=img.flatten()
        #透過スポットを計算から除外する場合
        #img=np.delete(img,list_index_DB)

        original.append(img.flatten())

    original_array=np.stack(original,axis=-1) #1次元配列化した回折図形の画素値を2次元にスタックした配列

    #画素を0~1に変換．こうした方が数値的に安定するらしい．
    original_array=original_array/255.


    base1=cv2.imread(bases[num_base_1-1],-1)
    base2=cv2.imread(bases[num_base_2-1],-1)
    base1=base1.flatten()
    base2=base2.flatten()
    #透過スポットを計算から除外する場合
    #base1=np.delete(base1,list_index_DB)
    #base2=np.delete(base2,list_index_DB)

    A=np.vstack((base1,base2)).transpose()
    y=original_array
    alpha=np.linalg.lstsq(A,y,rcond=None)[0] #MLSのメインの計算

    array_result=np.zeros_like(numbering)

    for i in range(len(alpha)): #基底の数だけ繰り返し処理
        coeff_map_1d=alpha[i]
        coeff_map_1d=contrast_opt(coeff_map_1d,max_value)

        num_coeff=0
        for y,x in index_array:
            array_result[y,x]=coeff_map_1d[num_coeff]
            num_coeff+=1

        name=str(list_num_base[i]).zfill(3)+".tif"
        cv2.imwrite(dir_save_map+name,array_result)
    num_for_GB_pixels+=1



print("fin.")