import glob


def write_name():
    # npz文件路径
    # files = glob.glob(r'D:\DEMO\SCUNet++\datasets\Synapse\train_npz\*.npz')
    files = glob.glob(r'G:\FINAL\SCUNet++\datasets\Synapse\test_vol_h5\*.npz')
    # txt文件路径
    # f = open(r'D:\DEMO\SCUNet++\lists\lists_Synapse\train.txt', 'w')
    f = open(r'G:\FINAL\SCUNet++\lists\lists_Synapse\test.txt', 'w')
    for i in files:
        name = i.split('\\')[-1]
        name = name[:-4] + '\n'
        f.write(name)

    print("Finished!")


write_name()
