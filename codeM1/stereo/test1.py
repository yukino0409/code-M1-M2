import numpy as np
import os
import cv2

# 粒界の3重点から抽出した2値化画像に対して、ステレオ再構成のピクセルマッチングに使用するエッジピクセルを抽出するコード（単一ごとに）
dir_save_stack = r"E:\Liu\20240405\stereo\edge_detection\\"
os.makedirs(dir_save_stack, exist_ok=True)
dir_save_0deg = r"E:\Liu\20240405\stereo\edge_detection\alpha_0deg\\"
os.makedirs(dir_save_0deg, exist_ok=True)
dir_save_20deg = r"E:\Liu\20240405\stereo\edge_detection\alpha_20deg\\"
os.makedirs(dir_save_20deg, exist_ok=True)

path_0deg = r"E:\Liu\20240405\add_results\alpbha_0deg\\"
path_20deg = r"E:\Liu\20240405\add_results\alpbha_20deg\\"

# 後半2枚はエッジ抽出はしない（粒界の連結部分において、どこまでを再構成の対象とする粒界なのかを決定するため）
list_file_name = ["001_002.tif", "002_003.tif", "002_004.tif", "003_004.tif"]

max_value = 255

# np.whereを用いてインデックスを取得する場合、この関数を通して2D配列に変換すると処理しやすい
def after_np_where(array):
    array = (np.array(array)).flatten()
    array = array.reshape([-1, 2], order='F')
    return array

list_img_0deg = []
list_img_20deg = []
for i in range(len(list_file_name)):
    img_0deg = cv2.imread(path_0deg + list_file_name[i], 0)
    img_20deg = cv2.imread(path_20deg + list_file_name[i], 0)
    list_img_0deg.append(img_0deg)
    list_img_20deg.append(img_20deg)

# 単一粒界の二値化画像をスタックして再度二値化
stack_0deg = (np.where(np.sum(np.array(list_img_0deg, dtype=np.int32), axis=0) >= max_value, max_value, 0)).astype(np.uint8)
stack_20deg = (np.where(np.sum(np.array(list_img_20deg, dtype=np.int32), axis=0) >= max_value, max_value, 0)).astype(np.uint8)

# 塗りつぶし
# 粒界のスタック画像からエッジを抽出
contours_0deg, _ = cv2.findContours(stack_0deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours_20deg, _ = cv2.findContours(stack_20deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
black_0deg = np.zeros_like(stack_0deg)
black_20deg = np.zeros_like(stack_20deg)

# スタック画像を画像出力（エッジ抽出がうまく行えているかの確認も含めて）
for p in contours_0deg:
    cv2.fillPoly(black_0deg, [p], max_value)
for p in contours_20deg:
    cv2.fillPoly(black_20deg, [p], max_value)
cv2.imwrite(dir_save_stack + "stack_alpha_0deg.tif", black_0deg)
cv2.imwrite(dir_save_stack + "stack_alpha_20deg.tif", black_20deg)

# スタック画像の輪郭抽出（単一粒界の輪郭ピクセルかつスタック画像の輪郭ピクセルをステレオ再構成に使用するピクセルとする）
edge_0deg = np.zeros_like(stack_0deg)
edge_20deg = np.zeros_like(stack_20deg)
contours, hierarchy = cv2.findContours(black_0deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
if len(contours) > 0:
    for x, y in contours[0][:, 0, :]:
        edge_0deg[y, x] = max_value
contours, hierarchy = cv2.findContours(black_20deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
if len(contours) > 0:
    for x, y in contours[0][:, 0, :]:
        edge_20deg[y, x] = max_value
# cv2.imwrite(dir_save_stack+"test.tif",edge_0deg)


for i in range(len(list_file_name) - 2):
    img_0deg = cv2.imread(path_0deg + list_file_name[i], 0)
    img_20deg = cv2.imread(path_20deg + list_file_name[i], 0)

    img_edge_0deg = np.zeros_like(img_0deg)
    img_edge_20deg = np.zeros_like(img_20deg)

    contours, hierarchy = cv2.findContours(img_0deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for x, y in contours[0][:, 0, :]:
            img_edge_0deg[y, x] = max_value
    contours, hierarchy = cv2.findContours(img_20deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for x, y in contours[0][:, 0, :]:
            img_edge_20deg[y, x] = max_value

    # 画像を重ね合わせて被っている個所を残す
    add_0deg = edge_0deg // max_value + img_edge_0deg // max_value
    add_20deg = edge_20deg // max_value + img_edge_20deg // max_value
    add_0deg = (np.where(add_0deg == 2, max_value, 0)).astype(np.uint8)
    add_20deg = (np.where(add_20deg == 2, max_value, 0)).astype(np.uint8)
    index_0deg = after_np_where(np.where(add_0deg == max_value))
    index_20deg = after_np_where(np.where(add_20deg == max_value))

    # ここのコードは消さない
    # 島（エラー）を消す
    """
    for y, x in index_0deg:
        yy = y - 1
        xx = x - 1
        if yy < 0:
            yy = 0
        if xx < 0:
            xx = 0
        if len(np.where(add_0deg[yy:y + 2, xx:x + 2] == max_value)[0]) == 1:
            add_0deg[y, x] = 0
    for y, x in index_20deg:
        yy = y - 1
        xx = x - 1
        if yy < 0:
            yy = 0
        if xx < 0:
            xx = 0
        if len(np.where(add_20deg[yy:y + 2, xx:x + 2] == max_value)[0]) == 1:
            add_20deg[y, x] = 0
            # print("iland")
    """

    """#ここのコードは消さない
    #画像の端のピクセルを削除
    add_0deg_copy=add_0deg.copy()
    add_20deg_copy=add_20deg.copy()
    for y,x in index_0deg:
        #y,x=0,6
        if y==0:
            xx=x-1
            if xx<0:
                xx=0
            #if len(np.where(add_0deg[x-1:x+2,y]==0)[0])==0:
            if len(np.where(add_0deg_copy[y,xx:x+2]==0)[0])==0:
                add_0deg[y,x]=0
        if x==0:
            yy=y-1
            if yy<0:
                yy=0
            if len(np.where(add_0deg_copy[yy:y+2,x]==0)[0])==0:
                add_0deg[y,x]=0

    for y,x in index_20deg:
        if y==0:
            xx=x-1
            if xx<0:
                xx=0
            if len(np.where(add_20deg_copy[y,xx:x+2]==0)[0])==0:
                add_20deg[y,x]=0
        if x==0:
            yy=y-1
            if yy<0:
                yy=0
            if len(np.where(add_20deg_copy[yy:y+2,x]==0)[0])==0:
                add_20deg[y,x]=0
    """
    # 端の黒埋め（コメントアウトしても良いが、厳密には画像の端1pixelは粒界のエッジピクセルとして議論（再構成）できないはず）
    trim_0deg = np.zeros_like(add_0deg)
    trim_20deg = np.zeros_like(add_20deg)
    # test=add_0deg[1:add_0deg.shape[0]-1,1:add_0deg.shape[1]-1]
    trim_0deg[1:add_0deg.shape[0] - 1, 1:add_0deg.shape[1] - 1] = add_0deg[1:add_0deg.shape[0] - 1, 1:add_0deg.shape[1] - 1]
    trim_20deg[1:add_20deg.shape[0] - 1, 1:add_20deg.shape[1] - 1] = add_20deg[1:add_20deg.shape[0] - 1, 1:add_20deg.shape[1] - 1]

    # エッジピクセルの左端or右端を決定

    # エッジピクセルの左端or右端を決定
    list_integral = []
    contours, _ = cv2.findContours(trim_0deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) >= 2:
        for j in range(2):
            integral = 0
            elements = 0
            for x, y in contours[j][:, 0, :]:
                integral += x
                elements += 1
            list_integral.append(integral / elements)
        num_right = list_integral.index(max(list_integral))
        for x, y in contours[num_right][:, 0, :]:
            trim_0deg[y, x] = max_value // 2  # 左端と区別するために、右端は画素値=255//2で出力

    list_integral = []
    contours, _ = cv2.findContours(trim_20deg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) >= 2:
        for j in range(2):
            integral = 0
            elements = 0
            for x, y in contours[j][:, 0, :]:
                integral += x
                elements += 1
            list_integral.append(integral / elements)
        num_right = list_integral.index(max(list_integral))
        for x, y in contours[num_right][:, 0, :]:
            trim_20deg[y, x] = max_value // 2

    cv2.imwrite(dir_save_0deg + list_file_name[i], trim_0deg.astype(np.uint8))
    cv2.imwrite(dir_save_20deg + list_file_name[i], trim_20deg.astype(np.uint8))

print("fin.")
