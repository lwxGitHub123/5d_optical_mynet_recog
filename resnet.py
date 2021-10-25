import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from MyDatasetCuda import MyDatasetCuda
import time


root="F:/liudongbo/dataSet/small_pic/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data=MyDatasetCuda(root=root,txt=root+'train.txt', transform=transforms.ToTensor())
#train_data = train_data.to(device)
test_data=MyDatasetCuda(root=root,txt=root+'test.txt', transform=transforms.ToTensor())
#test_data = test_data.to(device)
train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=32)


#-----------------create the Net and training------------------------

class Net(torch.nn.Module):
    def __init__(self):
        self.expansion = 4
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 7, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, 3, 2, 1),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 128, 3, 1, 1),
            torch.nn.ReLU()
        )
        self.conv5 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 3, 2, 1),
            torch.nn.ReLU()
        )
        self.conv6 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, 1, 1),
            torch.nn.ReLU()
        )
        self.conv7 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 512, 3, 2, 1),
            torch.nn.ReLU()
        )
        self.conv8 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, 3, 1, 1),
            torch.nn.ReLU()
        )
        self.pool = torch.nn.Sequential(
            torch.nn.AvgPool2d( 1, 1),
            torch.nn.ReLU()
        )

        self.fc = torch.nn.Sequential(

            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            #torch.nn.Dropout(p=0.5),
            torch.nn.Linear(64, 2)

        )

        # self.dropout = torch.nn.Dropout(p=0.5)


    def forward(self, x):

        printFlag = False

        if  printFlag == True:
            print("  x.shape =  ")
            print(x.shape)

        conv1_out = self.conv1(x)

        if printFlag == True:
            print("  conv1_out.shape =  ")
            print(conv1_out.shape)

        conv2_out = self.conv2(conv1_out)
        if printFlag == True:
            print("  conv2_out.shape =  ")
            print(conv2_out.shape)

        for i in range(5):

            conv2_out = self.conv2(conv2_out)
            if printFlag == True:
                print("  conv2_out.shape =  ")
                print(conv2_out.shape)
                print("  i =  ")
                print(i)

        conv3_out = self.conv3(conv2_out)
        if printFlag == True:
            print("  conv3_out.shape =  ")
            print(conv3_out.shape)

        conv4_out = self.conv4(conv3_out)
        if printFlag == True:
            print("  conv4_out.shape =  ")
            print(conv4_out.shape)

        for i in range(6):
            conv4_out = self.conv4(conv4_out)
            if printFlag == True:
                print("  conv4_out.shape =  ")
                print(conv4_out.shape)

        conv5_out = self.conv5(conv4_out)
        if printFlag == True:
            print("  conv5_out.shape =  ")
            print(conv5_out.shape)

        conv6_out = self.conv6(conv5_out)
        if printFlag == True:
            print("  conv6_out.shape =  ")
            print(conv6_out.shape)

        for i in range(10):
            conv6_out = self.conv6(conv6_out)
            if printFlag == True:
                print("  conv6_out.shape =  ")
                print(conv6_out.shape)
                print("  i = ")
                print(i)

        conv7_out = self.conv7(conv6_out)
        if printFlag == True:
            print("  conv7_out.shape =  ")
            print(conv7_out.shape)

        conv8_out = self.conv8(conv7_out)
        if printFlag == True:
            print("  conv8_out.shape =  ")
            print(conv8_out.shape)

        for i in range(4):
            conv8_out = self.conv8(conv8_out)
            if printFlag == True:
                print("  conv8_out.shape =  ")
                print(conv8_out.shape)


        pool = self.pool(conv8_out)
        if printFlag == True:
            print("  pool.shape =  ")
            print(pool.shape)

        view1 = conv1_out.view(pool.shape[0], -1)

        if printFlag == True:
            print("  view1.shape =  ")
            print(view1.shape)

        out = self.fc(view1)
        # out = torch.nn.Sequential(
        #
        #     torch.nn.Linear(conv3_out.shape[1], 4),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(4, 2)
        # )

        # out = torch.nn.Linear(8, 4)

        if printFlag == True:
            print("  out.shape =  ")
            print(out.shape)

        # out = self.dropout(out)

        return out

def train():

    model = Net()
    print(model)

    model.to(device) # 移动模型到cuda



    learning_rate = 0.0001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = torch.nn.CrossEntropyLoss()


    time_before = time.time()
    for epoch in range(300):
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
            print("  batch_xCount =  ")
            batch_xCount = batch_xCount + 1
            print(batch_xCount)


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

if __name__ == '__main__':
    train()


