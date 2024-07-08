import numpy as np
import cv2
import glob
import os
#隣り合う粒の全ての組み合わせをcsvファイルに出力するコード

#組み合わせをcsvファイルに出力し、保存するディレクトリ
dir_save=r"E:\Liu\20240405\MLS\\"
os.makedirs(dir_save,exist_ok=True)

#粒番号でナンバリングした画像を読み込み
numbering=cv2.imread(r"E:\Liu\20240405\mask_imgs\beta_20deg\whole\numbering.tif",0)
h,w=numbering.shape[:2]

index_GB=(np.array(np.where(numbering==0))).flatten()
index_GB=index_GB.reshape([-1,2],order='F')

list_combination=[]
#縦スキャン
for y,x in index_GB:
    if (x-1<0)or(x+1>w-1):
        continue
    else:
        target=numbering[y,x-1:x+2]
        numbers=np.unique(target)
        numbers=np.delete(numbers,np.where(numbers==0))

        if len(numbers)<2:
            continue
        else:
            numbers=tuple(numbers)
            list_combination.append(numbers)

#横スキャン
for y,x in index_GB:
    if (y-1<0)or(y+1>h-1):
        continue
    else:
        target=numbering[y-1:y+2,x]
        numbers=np.unique(target)
        numbers=np.delete(numbers,np.where(numbers==0))

        if len(numbers)<2:
            continue
        else:
            numbers=tuple(numbers)
            list_combination.append(numbers)

#順序が逆になっているだけで被りが生じている可能性はある(要コード修正)
result=set(list_combination)
result=list(result)
np.savetxt(dir_save+"grain_combination_beta_20deg.csv",result,delimiter=",")



print("fin.")