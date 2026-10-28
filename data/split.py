import os
import random

def split_dataset(image_folder, trainval_file, test_file, split_ratio=0.8):
    # 获取文件夹中的所有图片文件名
    all_images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    
    # 随机打乱文件名列表
    random.shuffle(all_images)

    # 计算分割索引
    split_index = int(len(all_images) * split_ratio)

    # 划分训练和测试集
    trainval_images = all_images[:split_index]
    test_images = all_images[split_index:]

    # 写入 trainval.txt
    with open(trainval_file, 'w') as f:
        for image in trainval_images:
            f.write(os.path.splitext(image)[0] + '\n')  # 写入不带后缀的文件名

    # 写入 test.txt
    with open(test_file, 'w') as f:
        for image in test_images:
            f.write(os.path.splitext(image)[0] + '\n')  # 写入不带后缀的文件名

# 使用示例
image_folder = '/root/分割IRSTD的benchmark/datasets/NUAA/images'  # 替换为你的图片文件夹路径
split_dataset(image_folder, '/root/分割IRSTD的benchmark/datasets/NUAA/idx_427/trainval.txt', '/root/分割IRSTD的benchmark/datasets/NUAA/idx_427/test.txt')