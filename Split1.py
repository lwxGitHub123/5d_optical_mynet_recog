import cv2
import numpy as np
import os


'''水平投影'''

def getHProjection(image):
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0] * h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 0:
                h_[y] += 1
    # 绘制水平投影图像
    # for y in range(h):
    #     for x in range(h_[y]):
    #         hProjection[y, x] = 255
    # cv2.imshow('hProjection2', hProjection)

    return h_


def getVProjection(image):
    vProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w) = image.shape
    #长度与图像宽度一致的数组
    w_ = [0]*w
    #循环统计每一列白色像素的个数
    for x in range(w):
        for y in range(h):
            if image[y,x] == 0:
                w_[x]+=1
    #绘制垂直平投影图像
    # for x in range(w):
    #     # for y in range(h-w_[x],h):
    #     for y in range( w_[x]):
    #         vProjection[y,x] = 255
    # cv2.imshow('vProjection',vProjection)
    return w_

def saveImg(img,xmin,ymin,xmax,ymax,imgIndex,SumThreshVal,img_p,px,save_path):

    type = img.shape

    x_centor = 0
    y_centor = 0

    # imgIndex = imgIndex + 1

    if len(type) == 3 :
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.namedWindow("gray1", cv2.WINDOW_NORMAL)
        # cv2.imshow('gray1', image)
        # 将图片二值化
        retval, img0 = cv2.threshold(image, SumThreshVal, 255, cv2.THRESH_BINARY)
        # cv2.namedWindow("binary1", cv2.WINDOW_NORMAL)
        # cv2.imshow('binary1', img0)
        roiImg = img0[ymin:ymax, xmin:xmax].copy()
        # cv2.namedWindow("roiImg", cv2.WINDOW_NORMAL)
        # cv2.imshow('roiImg', roiImg)
        x_start = 0
        x_end = 0
        y_start = 0
        y_end = 0
        colsum = np.sum(roiImg, axis=0)
        for i in range(len(colsum)):
            if colsum[i] > 0 :
                x_start = i + xmin
                print("  x_start =  ")
                print(x_start)
                print("  xmin = ")
                print(xmin)
                break

        countJ = 0
        for j in range(len(colsum) - 1, -1, -1):
            if colsum[j] > 0:
               x_end = xmax - countJ

               print("  x_end =  ")
               print(x_end)
               print("  xmax = ")
               print(xmax)
               break
            countJ = countJ + 1

        rowsum = np.sum(roiImg, axis=1)
        for i in range(len(rowsum)):
            if rowsum[i] > 0 :
                y_start = ymin + i
                print("  y_start =  ")
                print(y_start)
                print("  ymin = ")
                print(ymin)
                break

        countJ = 0
        for j in range(len(rowsum) - 1, -1, -1):
            if rowsum[j] > 0:
                y_end = ymax - countJ

                print("  y_end =  ")
                print(y_end)
                print("  ymax = ")
                print(ymax)
                break
            countJ = countJ + 1

        x_centor = int((x_start + x_end)/2)
        y_centor = int((y_start + y_end)/2)

        roiImg1 = img0[y_start:y_end, x_start:x_end]
        # cv2.namedWindow("roiImg1", cv2.WINDOW_NORMAL)
        # cv2.imshow('roiImg1', roiImg1)

        roiImg2 = img0[y_centor - 9:y_centor + 9, x_centor - 9:x_centor + 9]
        split_img_d = img[y_centor - px:y_centor + px, x_centor - px:x_centor + px]  # 先写y后写x,切图输入[height, width]
        pic_name_d = save_path + '/' + str(imgIndex) + '_d' + ".jpg"
        cv2.imwrite(pic_name_d, split_img_d)

        split_img_p = img_p[y_centor - px:y_centor + px, x_centor - px:x_centor + px]  # 先写y后写x,切图输入[height, width]
        pic_name_p = save_path + '/' + str(imgIndex) + '_p' + ".jpg"
        cv2.imwrite(pic_name_p, split_img_p)

        # cv2.namedWindow("roiImg2", cv2.WINDOW_NORMAL)
        # cv2.imshow('roiImg2', roiImg2)
        #
        #
        # cv2.waitKey(0)
    else :
        retval, binary = cv2.threshold(img, 205, 255, cv2.THRESH_BINARY_INV)
        roiImg = binary[ymin:ymax, xmin:xmax]
        cv2.namedWindow("roiImg", cv2.WINDOW_NORMAL)
        cv2.imshow('roiImg', roiImg)
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow('img', binary)
        cv2.waitKey(0)



def del_file(path):
    lsdir = os.listdir(path)
    print(lsdir)
    if any(name.endswith('.py') for name in lsdir):
       print("no txt in this dir")
    else:
      print("have txt and need to remove")

    for file in lsdir:
        try:
            c_path = os.path.join(path,file)
            os.remove(c_path)
            print("rm c path: %s " % c_path)
        except:
            #del_file(path)
            os.rmdir(c_path)
            print("rm failed try again: %s " % c_path)


def writeToTxt(content,directory,fileName):

    textPath = directory + "/" + str(fileName) + ".txt"
    f = open(textPath, 'a')

    # if imgindex % 57 == 0 :
    #     f.write('\n')
    f.write(str(content) + '\n')
    f.close()

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


def splitImg1(pic_path,pic_num,suffix,ThreshVal,SetExceptAreaVal,SetM,px,save_path,SumThreshVal,txtName,test_size):

    labels = GetLabels(txtName)

    smallImgNumber = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    BASE_PATH = os.getcwd()
    tmp_path = os.path.join(BASE_PATH, save_path)
    del_file(tmp_path)

    for hi in range(1, pic_num+1):
        img_name_d = pic_path + '/' + str(hi) + '_delay.' + suffix
        img_name_p = pic_path + '/' + str(hi) + '_plr.' + suffix
        img_d = cv2.imread(img_name_d)                                      # 只有delay的图能分割出轮廓
        img_p = cv2.imread(img_name_p)

        origineImage = img_d

        image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
        cv2.imshow('gray', image)
        # 将图片二值化
        retval, img0 = cv2.threshold(image, ThreshVal, 255, cv2.THRESH_BINARY_INV)


        cv2.namedWindow("binary", cv2.WINDOW_NORMAL)
        cv2.imshow('binary', img0)
        # 图像高与宽
        (h, w) = img0.shape
        img = img0[int(h/2)*0:int(h/1),:]

        origineimg = origineImage[int(h/2)*0 :int(h/1), :]

        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        # cv2.imshow('img', img)
        cv2.waitKey(0)

        Position = []
        # 水平投影
        H = getHProjection(img)

        print("  len(H) =  ")
        print(len(H))

        # cv2.waitKey(0)

        start = 0
        H_Start = []
        H_End = []
        # 根据水平投影获取垂直分割位置
        for i in range(len(H)):
            # print(" i = ")
            # print(i)
            # print("  H[i] = ")
            # print( H[i])

            if H[i] > 0 and start == 0:
                H_Start.append(i)
                start = 1
            if H[i] <= 0 and start == 1:
                H_End.append(i)
                start = 0

        print("   len(H_Start)  =   ")
        print(len(H_Start))
        print("  len(H_End) =  ")
        print(len(H_End))

        W_Start_Len =  -1
        # 分割行，分割之后再进行列分割并保存分割位置
        for i in range(len(H_Start)):
            # 获取行图像
            # cropImg = img[H_Start[i]:H_End[i], 0:w]
            # cropImgi = origineImage[H_Start[i]:H_End[i], 0:w]
            # cv2.imshow('cropImg' + str(i),cropImgi)
            # cv2.waitKey(0)

            # 对行图像进行垂直投影
            W = getVProjection(img)
            Wstart = 0
            Wend = 0
            W_Start = 0
            W_End = 0
            # print(" len(W) =  ")
            # print(len(W))
            for j in range(len(W)):
                # print(" W[j] == ")
                # print(W[j])
                # print(" j == ")
                # print(j)
                if W[j] > 0 and Wstart == 0:
                    W_Start = j
                    Wstart = 1
                    Wend = 0
                if W[j] <= 0 and Wstart == 1:
                    W_End = j
                    Wstart = 0
                    Wend = 1
                if Wend == 1:
                   if j % (len(W)) == 0 :
                      Position.append([W_Start, H_Start[i], W_End, H_End[i],1])
                   else :
                      Position.append([W_Start, H_Start[i], W_End, H_End[i], 0])
                   Wend = 0

            W_Len = len(W)

        # 根据确定的位置分割字符
        PositionList = []
        print(" len(Position) = ")
        print(len(Position))

        trainLen = len(Position) * (1 - test_size)
        if hi in SetExceptAreaVal:
           trainLen = (len(Position) - SetM) * (1 - test_size)

        for m in range(len(Position) - 1,-1,-1):

            # Position1 = sorted(Position, reverse=True)
            print("  m  =   ")
            print(m)


           # areaVal,Position[m][0], Position[m][1], Position[m][2], Position[m][3] = saveImg(origineImage,Position[m][0], Position[m][1], Position[m][2], Position[m][3],smallImgNumber,img_p,px,save_path)


            # AearaSumList.append(areaVal)
            if  hi in SetExceptAreaVal :

                if m > SetM :
                    smallImgNumber = smallImgNumber + 1
                    saveImg(origineImage, Position[m][0], Position[m][1], Position[m][2], Position[m][3], smallImgNumber,
                            SumThreshVal,img_p, px, save_path)

                    cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]), (0, 229, 238), 1)

            else :
                smallImgNumber = smallImgNumber + 1
                saveImg(origineImage, Position[m][0], Position[m][1], Position[m][2], Position[m][3], smallImgNumber,
                        SumThreshVal,img_p, px, save_path)

                cv2.rectangle(origineImage, (Position[m][0], Position[m][1]), (Position[m][2], Position[m][3]),
                              (0, 229, 238), 1)

            label = labels[smallImgNumber-1][0]
            content = str(smallImgNumber) + '_d'+ '.jpg' + '  ' + str(label)

            if m <= trainLen:
                fileName = "train"
                writeToTxt(content, save_path, fileName)
            else:
                fileName = "test"
                writeToTxt(content, save_path, fileName)

            # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
            # cv2.imshow('image', origineImage)
            # cv2.waitKey(0)


    return  smallImgNumber
