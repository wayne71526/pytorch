## 載入套件
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST


## 設定參數
path = ''   # 當前目錄
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用 GPU，若無 GPU，使用 CPU


## 載入資料
transform = transforms.Compose([
    transforms.ToTensor(),       # 轉換成 pytorch 張量，並進行歸一化(將圖像的像素值從 [0, 255] 的範圍轉換到 [0, 1] 範圍)
    transforms.Normalize(mean=(0.1307, ), std=(0.3081, ))   # 標準化  
])

train_data = MNIST(path, train=True, download=True, transform=transform)
test_data  = MNIST(path, train=False, download=True, transform=transform)


## 建立模型
class Mnist_CNN(nn.Module):
    def __init__(self, num_class=10):
        super(Mnist_CNN, self).__init__()  # 調用 torch.nn.Module 的構造函數
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),   # 28-5+2*2+1 = 28
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),   # Relu：用來增加網路的非線性特性
            nn.MaxPool2d(kernel_size=2, stride=2)         # (28-2)/2+1 = 14
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),  # 14
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),   # Relu：用來增加網路的非線性特性
            nn.MaxPool2d(kernel_size=2, stride=2)         # (14-2)/2+1 = 7
        )
        self.fc = nn.Linear(7*7*64, num_class)
        
    def forward(self, x):   # 調用實例時，會自動運行
        output = self.layer1(x)
        output = self.layer2(output)
        output = output.reshape(output.size(0), -1)       # out.size(0)：batch_size
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)             # 常和損失函數 nn.NLLLoss 一起使用
        return output 
    
model = Mnist_CNN().to(device)   # 將模型移動到指定設備


## 模型訓練
batch_size = 1000     # 多少個樣本訓練一次
epochs = 10           
lr = 0.1              # 學習率
train_loader = DataLoader(train_data, batch_size=batch_size)

# 建立優化器
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
loss_list = []
for epoch in range(1, epochs+1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()   # 梯度重置
        output = model(data)    # 調用模型實例
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()        # 更新權重
        
        if (batch_idx+1) % 10 == 0:
            loss_list.append(loss.item())
            total = len(train_loader.dataset)
            trained_num = (batch_idx+1)*batch_size
            percent = round(100*trained_num / total, 1)
            print(f'epoch {epoch}：{trained_num}/{total}({percent}%)，loss：{loss.item()}')


## 對損失繪圖
import matplotlib.pyplot as plt

plt.plot(loss_list, 'r')


## 對 test_data 做預測
test_loader = DataLoader(test_data, batch_size)

model.eval()
test_loss = 0
correct = 0

with torch.no_grad():  # 不計算梯度，減少內存消耗和計算開銷
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        test_loss += F.nll_loss(output, target)
        _, prediction = torch.max(output, dim=1)
        correct += (prediction == target).sum().item()
    
avg_loss = test_loss / len(test_loader.dataset)
accuracy = 100 * correct / len(test_loader.dataset)
print(f'Accuracy：{accuracy:.1f} %，Average Loss：{avg_loss.item():.4f}')


## 對 test data 前 20 筆資料作預測
predict = []

with torch.no_grad():  # 不計算梯度，減少內存消耗和計算開銷
    for i in range(20):
        data = test_data[i][0]
        data = data.reshape(1, *data.shape)   # 增加 1 維度：1 channel(灰階)

        output = model(data)
        pred = torch.argmax(output).item()    # 預測值
        predict.append(pred)
    
print('true target：', test_data.targets[:20].tolist())
print('prediction： ', predict)