import hyperspy.api as hs
import numpy as np
import cv2
import os
#blockファイルから回折図形を出力するコード

dir_save=r"E:\Liu\20240405\diff_img_raw\beta_20deg\\" #回折図形の画像を出力するディレクトリを指定
os.makedirs(dir_save,exist_ok=True) #上記で指定したディレクトリを作成するwo

data_block=hs.load(r"E:\Liu\20240405\20240405_beta_20deg.blo") #blockファイルを読み込む
data_block=data_block.data.transpose(0,1,2,3) #配列の軸の入れ替え←軸の入れ替えはしていないがこれがないとエラー出た
h_vbf,w_vbf,h_dif,w_dif=data_block.shape #4Dデータの配列の形状を取得

for y in range(h_vbf):
    for x in range(w_vbf):
        diff_pattern=data_block[y][x] #実空間の座標を指定し、その座標の回折図形を抽出する

        """
        #オーバーフロー対策
        img_min=np.amin(diff_patterns)
        img_max=np.amax(diff_patterns)
        diff_patterns=diff_patterns-img_min #img_minの正負にかかわらず最小値を0にする
        img_min=np.amin(diff_patterns)
        img_max=np.amax(diff_patterns)
        diff_patterns=diff_patterns/img_max*65535 #画素値を0-65535に拡張
        diff_patterns=np.asarray(diff_patterns,np.uint16) #uint8→uint16
        """
        name=str(y).zfill(3)+"_"+str(x).zfill(3)+".tif" #画像ファイル名を指定
        cv2.imwrite(dir_save+name,diff_pattern.astype(np.uint8)) #画像を指定のディレクトリに出力



print("fin.")