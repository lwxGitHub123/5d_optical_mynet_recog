import torch
import torch.nn as nn
from Model import Model
import os
import cv2
from Split import splitImg


# 定义数据
# x:输入数据
# y:标签

def GetLabels(txt):

    labels = []
    fh = open(txt, 'r')
    imgs = []
    for line in fh:
        line = line.strip('\n')
        line = line.rstrip()
        words = line.split()
        print("  line =  ")
        print(line)
        phase = words[3]
        color = words[4]
        temp = [phase,color]
        labels.append(temp)

    return labels

def MyDataSets(imgPath0,labels):

    AllPositionList = []
    ThreshVal =200
    positionIndex= 0

    for filename in os.listdir(imgPath0):
        print(filename)
        imgPath = imgPath0 + filename

        # 读入原始图像
        origineImage = cv2.imread(imgPath)

        (h, w,_) = origineImage.shape
        img = origineImage[0:int(h / 1), :]
        AllPositionList = splitImg(img, positionIndex, ThreshVal,56,labels)
        print(" len(AllPositionList) = ")
        print(len(AllPositionList))




if __name__ == '__main__':



    # txt = "20210612-5.txt"
    txtPath = "F:/liudongbo/dataSet/20210809-1.txt"
    labels = GetLabels(txtPath)
    print("  labels  =  ")
    print(labels)

    imgFile = "F:/liudongbo/dataSet/bmp/20210809-1/"

    MyDataSets(imgFile,labels)

    x = torch.Tensor([[0.2, 0.4], [0.2, 0.3], [0.3, 0.4],[0.9,0.6]])
    y = torch.Tensor([[0.6], [0.5], [0.7],[0.8]])



    net = Model()
    Epoch = 10000

    for i in range(Epoch):
        # print("  i = ")
        # print(i)
        # print(" x = ")
        # print(x)
        # print("  y = ")
        # print(y)
        net.train(x, y)


    PATH = "F:/liudongbo/projects/net_params.pkl"
    torch.save(net, PATH)

    x1 = torch.Tensor([[0.9, 0.6]])
    out = net.test(x1)
    print(out)  # 输出结果 tensor([[0.5205]], grad_fn=<AddmmBackward>)



