import numpy as np
import cv2
import glob
import os


def npz():
    # 原图像存放的路径
    # path = r'D:\DEMO\SCUNet++\datasets\Synapse\train\images\*.png'
    path = r'G:\FINAL\SCUNet++\datasets\test\images\*.png'
    # 存放训练（测试）所用的npz文件路径
    # path2 = r'D:\Research_Topic\Swin-Unet-main\datasets\Synapse\train_npz\\'
    # path2 = r'D:\DEMO\SCUNet++\datasets\Synapse\train_npz'
    path2 = r'G:\FINAL\SCUNet++\datasets/Synapse/test_vol_h5'
    for i, img_path in enumerate(glob.glob(path)):
        # 读入图像
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 读入标签
        label_path = img_path.replace('images', 'labels')
        label = cv2.imread(label_path, flags=0)
        # 将非目标像素设置为0
        label[label == 0] = 0
        # 将目标像素设置为1
        label[label != 0] = 1
        # 保存npz文件
        # print(os.path.join(path2, str(i + 1)))
        np.savez(os.path.join(path2, str(i + 1)), image=image, label=label)
        print('finished:', i + 1)

    print('Finished')


npz()
