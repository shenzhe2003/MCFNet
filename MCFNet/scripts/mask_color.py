import cv2
import numpy as np
import os

def mask_white_to_other(
    mask_path,
    save_path,
    white_thresh=200,
    color=(255, 100, 10)  # ★ 不那么亮的蓝色（BGR）
    # color = (30, 60, 180)   # 深红，和 (180,60,30) 是对称色


):
    """
    将掩码图中白色区域变成蓝色，生成彩色图片
    """
    # 读取为灰度
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read image: {mask_path}")

    h, w = mask.shape

    # 创建 RGB 图像（BGR for cv2）
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    # 判断“白色掩码区域”
    mask_region = mask >= white_thresh

    color_img[mask_region] = color

    # 保存
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    cv2.imwrite(save_path, color_img)


if __name__ == "__main__":
    in_mask  = "image.png"          # 输入掩码
    out_img  = "blue_53.png"     # 输出结果

    mask_white_to_other(in_mask, out_img)
    print("Saved:", out_img)
