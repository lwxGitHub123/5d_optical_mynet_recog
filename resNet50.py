import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from MyDatasetCuda import MyDatasetCuda
import time
from resnetModel  import  resnet18
from visualize import visualize
from gen_dataset import gen_dataset
from txt2csv_new import txt2csv
from split0903 import splitImg




root="G:/liudongbo/dataset/small_pic_test/"

SAVE_PATH = 'G:/liudongbo/dataset/small_pic_test/'             # 切出来的小图保存路径

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data=MyDatasetCuda(root=root,txt=root+'train.txt', transform=transforms.ToTensor())
#train_data = train_data.to(device)
test_data=MyDatasetCuda(root=root,txt=root+'test.txt', transform=transforms.ToTensor())
#test_data = test_data.to(device)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True,pin_memory=True ,num_workers = 8)
test_loader = DataLoader(dataset=test_data, batch_size=64,pin_memory=True ,num_workers = 8)

# all_data=MyDatasetCuda(root=root,txt=root+'all.txt', transform=transforms.ToTensor())
# all_loader = DataLoader(dataset=all_data, batch_size=2080, shuffle=True,pin_memory=True ,num_workers = 12)
# print("  ")


def train(Epoch,Learning_rate):

    model = resnet18()
    print(model)

    model.to(device) # 移动模型到cuda



    learning_rate = Learning_rate

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()


    time_before = time.time()

    for epoch in range(Epoch):
      # with torch.no_grad():
        print('epoch {}'.format(epoch + 1))
        batch_xCount = 0
        # training-----------------------------
        train_loss = 0.
        train_acc = 0.
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 有GPU则将数据置入GPU加速

            # print("  batch_x.shape =  ")
            # print(batch_x.shape)

            batch_xCount = batch_xCount + 1
            if  batch_xCount % 1000 == 0 :

                print('batch_xCount {}'.format(batch_xCount))


            out = model(batch_x)
            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            # pred = torch.max(out, 1)[1]
            pred = out.argmax(dim=1)

            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
            train_data)), train_acc / (len(train_data))))

        # evaluation--------------------------------
        testNum = 0
        model.eval()
        eval_loss = 0.
        eval_acc = 0.
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 有GPU则将数据置入GPU加速

            testNum = testNum + 1

            out = model(batch_x)
            loss = loss_func(out, batch_y)
            eval_loss += loss.item()
            # pred = torch.max(out, 1)[1]
            pred = out.argmax(dim=1)
            # print("  pred.shape = ")
            # print(pred.shape)
            # print(" batch_y.shape = ")
            # print(batch_y.shape)
            # print("  testNum =  ")
            # print(testNum)
            num_correct = (pred == batch_y).sum()
            eval_acc += num_correct.item()
        print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
            test_data)), eval_acc / (len(test_data))))

    time_after = time.time()
    print('totalTime: {:.6f}'.format(time_after - time_before))

    return  model

def getImgNameAndLabels(filename):

    imgNameList = []
    labelList = []
    with open(filename, "r") as f:  # 打开文件

        # print(data)
        for line in f:
            # print(" line  =  ")
            # print(line)
            lineList = line.split('  ')
            imgName = lineList[0]
            label = lineList[1]
            imgNameList.append(imgName)
            labelList.append(label)

    return imgNameList , labelList

if __name__ == '__main__':


    ################ 生成训练集超参数 ##################
    TYPE = 'both'  # 要训练的模型类型
    DELAY = [80, 120]  # [50, 100]
    PLR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
           15]  # [0,5,10, 15, 20, 25,30,35,40,45,50,55,60,65,70,75] #  #[5,20,35,50]# #   [27,36,45,54]
    INDEX = ['x', 'y', 'z', 'delay', 'plr']
    ################# 切割图片超参数 ###################

    EPOCH = 30000
    learning_rate = 0.00001


    ################ 生成训练集超参数 ##################
    ################### 训练超参数 ####################
    if TYPE == 'delay':
        N_O = len(DELAY)
    elif TYPE == 'plr':
        N_O = len(PLR)
    elif TYPE == 'both':
        N_O = len(PLR) * len(DELAY)





    model = train(Epoch = EPOCH,Learning_rate = learning_rate)

    IMGNAMELIST, LABELLIST = getImgNameAndLabels(root + "/all.txt")

    PLOT_ONLY = len(LABELLIST)  # 画多少个点在聚类图上


    pic_tensor, labels = gen_dataset(pic_path=SAVE_PATH,imgNameList = IMGNAMELIST, labelList = LABELLIST)



    print('\n===================== 可视化聚类中 ===========================')
    # with torch.no_grad():
    #     for batch_x, batch_y in all_loader:
    #         batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
    #         batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 有GPU则将数据置入GPU加速

    pic_tensor = pic_tensor.to(device)
    visualize(model, input=pic_tensor, label=labels, plot_only=PLOT_ONLY, n_labels=N_O, title='4-bits')  # title改成对应的bits



