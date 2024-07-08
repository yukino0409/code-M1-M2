import numpy as np
import cv2
import glob
import os

#アライメントコード

dir_save_0deg=r"E:\Liu\20240226_stereo_with_sato\stereo\alignment\0deg\\"
os.makedirs(dir_save_0deg,exist_ok=True)
dir_save_20deg=r"E:\Liu\20240226_stereo_with_sato\stereo\alignment\20deg\\"
os.makedirs(dir_save_20deg,exist_ok=True)

imgs_0deg=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\edge_detection\0deg\*tif")
imgs_20deg=glob.glob(r"E:\Liu\20240226_stereo_with_sato\stereo\edge_detection\20deg\*tif")

#位置合わせの座標を入力する

"""
#アライメントの確認用（グレースケールで読み込む）
img_astar_0deg=cv2.imread(r"E:\Sato\20231016_SPED_austeniticsteel_for_stereo\MLS\numbering_map\GB_numbering\0deg\color\all\all_0deg.tif",-1)
img_astar_20deg=cv2.imread(r"E:\Sato\20231016_SPED_austeniticsteel_for_stereo\MLS\numbering_map\GB_numbering\20deg\color\all\all_20deg.tif",-1)
img_astar_0deg=cv2.cvtColor(img_astar_0deg,cv2.COLOR_BGR2GRAY)
img_astar_20deg=cv2.cvtColor(img_astar_20deg,cv2.COLOR_BGR2GRAY)
"""
name_list=[2,3,1]

black_0deg=np.zeros((123,109),dtype=np.uint8)
black_20deg=np.zeros((123,109),dtype=np.uint8)

"""
black_0deg_astar=np.zeros((123,109),dtype=np.uint8)
black_20deg_astar=np.zeros((123,109),dtype=np.uint8)
black_0deg_astar[0:120,21:21+88]=img_astar_0deg[0:120,0:88]
black_20deg_astar[2:2+121,0:109]=img_astar_20deg[0:121,0:109]
os.makedirs(dir_save_0deg+"astar",exist_ok=True)
os.makedirs(dir_save_20deg+"astar",exist_ok=True)
cv2.imwrite(dir_save_0deg+r"astar\0deg.tif",black_0deg_astar.astype(np.uint8))
cv2.imwrite(dir_save_20deg+r"astar\20deg.tif",black_20deg_astar.astype(np.uint8))
"""

for i in range(3):
    img_0deg=cv2.imread(imgs_0deg[i],0)
    img_20deg=cv2.imread(imgs_20deg[i],0)
    black_0deg[0:120,21:21+88]=img_0deg[0:120,0:88] #アライメントのため画像を切り取って黒画像に貼る
    black_20deg[2:2+121,0:109]=img_20deg[0:121,0:109]
    

    name=str(name_list[i]).zfill(3)+".tif"
    cv2.imwrite(dir_save_0deg+name,black_0deg.astype(np.uint8))
    cv2.imwrite(dir_save_20deg+name,black_20deg.astype(np.uint8))


print("fin.")