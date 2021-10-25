# 生成训练集
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn.model_selection import train_test_split    # sklearn内置分割训练集
import pandas as pd
import torchvision.transforms as transforms # (H, W, C) -> (C, H, W)
import torch
import cv2
import numpy as np
from PIL import Image
from torch.autograd import Variable
from common import writeToTxt



def getImgNameAndLabels(filename):

    imgNameList = []
    labelList = []
    with open(filename, "r") as f:  # 打开文件

        # print(data)
        for line in f:
            print(" line  =  ")
            print(line)
            lineList = line.split('  ')
            imgName = lineList[0]
            label = lineList[1]
            imgNameList.append(imgName)
            labelList.append(label)



# 图 -> tensor
def pic2tensor(pic_path, pic_num,imgNumList):

    imgNumListSum = []
    tmpNum = 0
    for i in range(len(imgNumList)):
        tmpNum = tmpNum + imgNumList[i]
        imgNumListSum.append(tmpNum)


    # pic_path为小图存放的文件夹名, pic_num为小图数量
    delay = []
    plr = []
    decode = []
    delay_tmp = []
    plr_tmp = []
    transf = transforms.ToTensor()
    for i in range(1, pic_num+1):
        img_d = cv2.imread(pic_path + '/%s_d.jpg' % str(i))
        img_p = cv2.imread(pic_path + '/%s_p.jpg' % str(i))

        img_d = img_d.reshape((img_d.shape[2],img_d.shape[0],img_d.shape[1]))
        img_p = img_p.reshape((img_p.shape[2], img_p.shape[0], img_p.shape[1]))

        # print("  img_d.shape =  ")
        # print(img_d.shape)

        print("  i =   ")
        print(i)

        delay.append(img_d)  # axis=0: 竖着堆叠
        plr.append(img_p)  # axis=0: 竖着堆叠

        decode.append(img_p)  # axis=0: 竖着堆叠
        plr_tmp.append(img_p)  # axis=0: 竖着堆叠



    delay_tensor = torch.tensor(delay)
    delay_tensor = Variable(delay_tensor.float())
    plr_tensor = torch.tensor(plr)
    plr_tensor = Variable(plr_tensor.float())
    decode_tensor = torch.tensor(decode)
    decode_tensor = Variable(decode_tensor.float())

    delay_m = decode
    delay_m_tensor = torch.tensor(delay_m)
    delay_m_tensor = Variable(delay_m_tensor.float())


    return delay, plr ,decode ,delay_tensor, plr_tensor ,decode_tensor

# 标签 -> tensor
def csv2tensor(filename,imgNumList,divVal):
    # filename为标签csv文件名
    data = pd.read_csv(filename)
    delay_ = data['delay']
    plr_ = data['plr']
    decode_ = data['decode']

    delay = []
    plr = []
    decode = []
    for i in range(len(delay_)):
        if i % divVal == 0 :
            delay.append(delay_[i])
            plr.append(plr_[i])
            decode.append(decode_[i])

    print("delay = ")
    print(delay)
    # for i in range(len(delay)):
    #     print(" delay[i] = ")
    #     print(delay[i])


    plr_list = []
    delay_list = []
    eachImgNum = 0
    if  len(imgNumList) <= 15 :
        for i in range(len(imgNumList)):

            eachImgNum = eachImgNum + imgNumList[i]
            start = 2 * eachImgNum - imgNumList[i]
            end = 2 * eachImgNum
            print(" i = ")
            print(i)
            print(" start* = ")
            print(start)
            print(" end = ")
            print(end)
            for j in range(start,end):
                plr_list.append(plr[j])
                # writeToTxt(str(plr[j]),"log","log")
    else :
        for i in range(0,len(imgNumList)):

            eachImgNum = eachImgNum + imgNumList[i]
            if i <= 15:
                start = 2 * eachImgNum - imgNumList[i]
                end = 2 * eachImgNum
                print(" i = ")
                print(i)
                print(" start* = ")
                print(start)
                print(" end = ")
                print(end)
                for j in range(start, end):
                    plr_list.append(plr[j])



    for h in range(eachImgNum):
        delay_list.append(delay[h])
        writeToTxt(str(delay[h]), "log", "log")


    print("plr = ")
    print(plr)

    # print("decode.shape = ")
    # print(decode.shape)





    # delay = delay.to_numpy()            # df -> np -> tensor
    delay = np.array(delay)

    # delay = torch.tensor(delay)
    delay_tensor = torch.tensor(delay)
    delay_tensor = Variable(delay_tensor)

    delay_list = np.array(delay_list)
    delay_list_tensor = torch.tensor(delay_list)
    delay_list_tensor = Variable(delay_list_tensor)



    # plr = plr.to_numpy()
    plr = np.array(plr)
    # plr = torch.tensor(plr)
    plr_tensor = torch.tensor(plr)
    plr_tensor = Variable(plr_tensor)

    # plr_list = plr_list.to_numpy()
    plr_list = np.array(plr_list)
    plr_list_tensor = torch.tensor(plr_list)
    plr_list_tensor = Variable(plr_list_tensor)

    print(" eachImgNum = ")
    print(eachImgNum)

    decode_list = []
    for h in range(eachImgNum):
        decode_list.append(decode[h])

        writeToTxt(" decode_list = " + str(decode_list[h]) , "log",
                   "log")
    # decode = decode.to_numpy()
    decode = np.array(decode)
    # decode = torch.tensor(decode)
    decode_tensor = torch.tensor(decode)
    decode_tensor = Variable(decode_tensor)

    decode_list = np.array(decode_list)
    decode_list_tensor = torch.tensor(decode_list)
    decode_list_tensor = Variable(decode_list_tensor)

    return delay, plr, decode, delay_list_tensor,plr_list_tensor, decode_list_tensor         # 返回标签tensor

def picandlabel2tensor(pic_path,imgNameList,labelNameList):


    imgList = []
    for i in range(len(imgNameList)):
        imgName = imgNameList[i]
        img = cv2.imread(pic_path + "/" + imgName)
        img = img.reshape((img.shape[2], img.shape[0], img.shape[1]))
        path = pic_path + "/" + imgName
        # img = Image.open(path).convert('RGB')
        imgList.append(img)

    # imgList_tensor = Variable(imgList)
    # imgList = np.array(imgList)

    imgList_tensor = torch.tensor(imgList)
    imgList_tensor = Variable(imgList_tensor.float())

    labelList = []
    for j in range(len(labelNameList)):
        label = int(labelNameList[j])
        labelList.append(label)

    label_list = np.array(labelList)
    label_list_tensor = torch.tensor(label_list)
    label_list_tensor = Variable(label_list_tensor)


    return  imgList_tensor ,label_list_tensor

# 生成数据集
def gen_dataset(pic_path, imgNameList, labelList):

    plc_tensor ,  label_tensor =picandlabel2tensor(pic_path, imgNameList, labelList)

    print("  plc_tensor.shape =  ")
    print(plc_tensor.shape)
    print("  label_tensor.shape = ")
    print(label_tensor.shape)


    return plc_tensor ,  label_tensor