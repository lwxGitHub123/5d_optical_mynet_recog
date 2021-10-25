import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from MyDataset import MyDataset
import time


root="./20210809-1/"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data=MyDataset(root=root,txt=root+'train.txt', transform=transforms.ToTensor())
#train_data = train_data.to(device)
test_data=MyDataset(root=root,txt=root+'test.txt', transform=transforms.ToTensor())
#test_data = test_data.to(device)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)


#-----------------create the Net and training------------------------

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2,0)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(6, 16, 5, 1, 0),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d( 2, 2, 0)
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(64, 120),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(120, 84),
            torch.nn.ReLU()
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(84, 2)
        )

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

        view1 = conv2_out.view(conv2_out.size(0), -1)
        if printFlag == True:
            print("  view1.shape =  ")
            print(view1.shape)

        fc1_out = self.fc1(view1)

        if printFlag == True:
            print("  fc1_out.shape =  ")
            print(fc1_out.shape)

        fc2_out = self.fc2(fc1_out)
        if printFlag == True:
            print("  fc2_out.shape =  ")
            print(fc2_out.shape)
        fc3_out = self.fc3(fc2_out)

        if printFlag == True:
            print("  fc3_out.shape =  ")
            print(fc3_out.shape)

        return fc3_out


model = Net()
print(model)



model.to(device) # 移动模型到cuda



learning_rate = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_func = torch.nn.CrossEntropyLoss()

time_before = time.time()
for epoch in range(300):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 有GPU则将数据置入GPU加速


        out = model(batch_x)
        loss = loss_func(out, batch_y)
        train_loss += loss.item()
        pred = torch.max(out, 1)[1]
        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_data)), train_acc / (len(train_data))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # 有GPU则将数据置入GPU加速

        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_data)), eval_acc / (len(test_data))))


time_after = time.time()
print('totalTime: {:.6f}'.format(time_after - time_before))

