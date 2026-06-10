import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from matplotlib.colors import ListedColormap

# 加载图片并转换为灰度图
def load_image_to_grayscale(image_path):
    image = Image.open(image_path).convert('L')
    return np.array(image)

# 3D可视化并保存
def visualize_image_in_3d(image_array, save_path):
    
    # 只对mask处理
    image_array = np.where(image_array < 20, 0, image_array)
    
    # 创建x和y的网格
    x = np.linspace(0, image_array.shape[0] - 1, image_array.shape[0])
    y = np.linspace(0, image_array.shape[1] - 1, image_array.shape[1])
    x, y = np.meshgrid(x, y)

    # 创建3D图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 自定义红-灰-蓝配色
    colors = ['darkred', 'blue', 'grey']
    cmap = ListedColormap(colors)
    # 绘制3D表面图
    ax.plot_surface(x, y, image_array, cmap='seismic', rstride=1, cstride=1)
    ax.view_init(elev=30, azim=45)  # 调整视图角度

    # 设置标签
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')

    # 保存图像到指定路径
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)  # 关闭图形，避免内存浪费

# 主函数：读取文件夹下的所有图片并处理
def main(input_folder, output_folder):
    # 如果输出文件夹不存在，则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_folder, filename)
            save_path = os.path.join(output_folder, f"3D_{filename}")

            print(f"Processing {image_path}...")

            # 加载图片并转换为灰度图
            image_array = load_image_to_grayscale(image_path)

            # 生成3D可视化图并保存
            visualize_image_in_3d(image_array, save_path)

    print("所有图片的3D可视化处理已完成！")

# 调用主函数
if __name__ == "__main__":
    input_folder = ''
    output_folder = ''
    main(input_folder, output_folder)
