import numpy as np
import cv2
import os
#ASTARの結晶方位マッピング画像から各粒のマスク画像を生成するコード
#（注）関数cv2.findContoursを用いて積層欠陥である小島の輪郭を抽出することは困難であるため、その部分はマッピングデータに応じて書き換える必要がある

#マスク画像の出力先
dir_save=r"E:\Liu\20240405\mask_imgs\alpha_0deg\\"
os.makedirs(dir_save,exist_ok=True)
#ASTARのマッピング画像(グレースケール,粒内:白,粒界:黒)の保存先
dir_save_gray=dir_save+"whole\\"
os.makedirs(dir_save_gray,exist_ok=True)

#ASTARの方位マッピング画像を読み込む
mapping=cv2.imread(r"E:\Liu\20240405\alpha_0deg\map_alpha_0deg_selected.tif")

#各種パラメータ設定
tilt_angle=0 #ファイル名をつけるためだけに使用する
max_value=255
mapping=mapping[0:82,0:115] #処理（粒界抽出,再構成）する領域を設定
mapping=cv2.cvtColor(mapping,cv2.COLOR_BGR2GRAY) #グレースケール(8bit)に変換する
mapping=np.where(mapping==0,0,max_value) #粒内:白,粒界:黒
mapping=mapping.astype(np.uint8)
cv2.imwrite(dir_save_gray+"map_"+str(tilt_angle)+"deg_gray.tif",mapping)
h,w=mapping.shape[:2]
#島(積層欠陥)の座標を入力←関数cv2.findContoursでは小島のような小さすぎる領域に対しては輪郭抽出精度が低下するため、別途（マニュアルで）処理した
y_island_offset=0
x_island_offset=0
h_island=0
w_island=0
island=mapping[y_island_offset:y_island_offset+h_island,x_island_offset:x_island_offset+w_island] #島になっている粒の領域



#輪郭抽出の関数
def find_contours(array,h,w):
    index_GB=np.array(np.where(array==0)).flatten()
    index_GB=index_GB.reshape([-1,2],order='F')
    for y_GB,x_GB in index_GB:
        if x_GB-1>=0:
            array[y_GB,x_GB-1]=0 #粒界ピクセルの左横のピクセルが粒界/粒内ピクセルに関わらず黒にする(輪郭抽出の関数を使用するため，輪郭抽出後に戻す)

    contours,hierarchy=cv2.findContours(array,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    list_results=[]
    for i in range(len(contours)):
        grain=np.zeros_like(array)
        grain=cv2.drawContours(grain,[contours[i]],0,max_value,-1) #-1:塗りつぶし

        grain_index=(np.array(np.where(grain>0))).flatten() #0:y,1:x
        grain_index=grain_index.reshape([-1,2],order='F') #[y,x]

        for y_G,x_G in grain_index:
            if x_G+1<=w-1:
                grain[y_G,x_G+1]=max_value

        list_results.append(grain)
        
    array_results=np.array(list_results,dtype=np.uint8)
    return array_results


#島になっている粒は別途先に処理する
#ここのコードはASTARのマッピング結果によって書き換える必要がある
array_result_island=np.zeros_like(island) #小島の輪郭抽出結果を張り付ける黒画像を用意する
num_y=0
for island_slice in island: #島の領域を1行ごとに処理していく

    index_GB_island=np.where(island_slice==0) #行内の粒界ピクセル(ASTAR)のインデックスを取得する
    index_GB_island=index_GB_island[0]

    if len(index_GB_island)>=2:
        list_difference=[]
        b=[index_GB_island[x+1]-index_GB_island[x] for x in range(len(index_GB_island)-1)]
        if set(b)=={1}:
            num_y+=1
            continue
        else:
            for a in range(len(index_GB_island)-1):
                b=index_GB_island[a]
                c=index_GB_island[a+1]
                d=c-b
                list_difference.append([b,c,d])
            array_difference=np.array(list_difference)
            index_diff_min=np.argmax(array_difference[:,2])
            left=array_difference[index_diff_min,0]
            right=array_difference[index_diff_min,1]
            array_result_island[num_y,left+1:right]=max_value
           
    num_y+=1

island_true=np.zeros_like(mapping)
island_true[y_island_offset:y_island_offset+h_island,x_island_offset:x_island_offset+w_island]=array_result_island
cv2.imwrite(dir_save+"001.tif",island_true.astype(np.uint8))


#島以外の粒の輪郭抽出
index_inside=np.where(array_result_island==max_value)
index_outside=np.where(island==0)
index_both=(np.hstack((index_inside,index_outside))).flatten()
index_both=index_both.reshape([-1,2],order='F')

mapping[y_island_offset:y_island_offset+h_island,x_island_offset:x_island_offset+w_island]=max_value #島を塗りつぶし
array_grains=find_contours(mapping,h,w)

num=2 #2スタート
for grain in array_grains:
    for y,x in index_both:
        grain[y+y_island_offset,x+x_island_offset]=0

    name=str(num).zfill(3)+".tif"
    cv2.imwrite(dir_save+name,grain.astype(np.uint8))
    num+=1



print("fin.")