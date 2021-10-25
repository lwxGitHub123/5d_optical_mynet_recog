import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from MyDatasetCuda import MyDatasetCuda
import time
from VggNet  import  VGG19


root="G:/liudongbo/dataset/small_pic_test/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data=MyDatasetCuda(root=root,txt=root+'train.txt', transform=transforms.ToTensor())
#train_data = train_data.to(device)
test_data=MyDatasetCuda(root=root,txt=root+'test.txt', transform=transforms.ToTensor())
#test_data = test_data.to(device)
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=8)



def train():

    model = VGG19(num_classes=1000)
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

            print("  batch_x.shape =  ")
            print(batch_x.shape)
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


