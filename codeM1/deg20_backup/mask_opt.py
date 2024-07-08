import numpy as np
import cv2
import os
#ASTARの結晶方位マッピング画像から各粒のマスク画像を生成するコード

#マスク画像の出力先
dir_save=r"E:\Liu\20240226_stereo_with_sato\mask_imgs\20deg\opt\\"
os.makedirs(dir_save,exist_ok=True)

#最適化したいマスク画像を読み込む
mask=cv2.imread(r"E:\Liu\20240226_stereo_with_sato\mask_imgs\20deg\opt\005.tif",0)

#各種パラメータ設定
max_value=255

mask[108:108+5,125:125+6]=max_value
mask[107,126:126+2]=max_value

cv2.imwrite(dir_save+"005.tif",mask.astype(np.uint8))

print("fin.")