import torch
from torch import nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd


## 讀取資料
train_data = pd.read_csv('ag_news_train.csv')
train_data['sentence'] = train_data['Title'] + ' ' + train_data['Description']

train_ds = []
for i in range(train_data.shape[0]):
    train_ds.append((train_data['Class Index'][i], train_data['sentence'][i]))

test_data = pd.read_csv('ag_news_test.csv')
test_data['sentence'] = test_data['Title'] + ' ' + test_data['Description']

test_ds = []
for i in range(test_data.shape[0]):
    test_ds.append((test_data['Class Index'][i], test_data['sentence'][i]))

train_iter = iter(train_ds)


## 詞彙表處理
tokenizer = get_tokenizer('basic_english') # 分詞

# 定義數據迭代器
def yield_tokens(data_iter):
  for _, text in data_iter:
    yield tokenizer(text)

# 建立詞彙表
voc = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"]) # unk 為 unknown 的縮寫，用於標示未在詞彙表的詞彙
voc.set_default_index(voc['<unk>']) # 查找不在詞彙表的詞時，會返回索引0

# 測試詞彙字典
print(voc(['here', 'is', 'an', 'example']))
print(voc(['安']))


## 定義資料轉換函數
def text_trans(text):
  return voc(tokenizer(text))   # 分詞且得到索引值

def label_trans(idx):
  return int(idx)-1


 ## 參數設定
voc_size = len(voc)
embed_size = 64
num_class = len(set([label for label, text in train_ds]))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 5
batch_size = 64
epoches = 10
print(num_class)
print(device)


## 建立模型
class TextClassfication(nn.Module):
  def __init__(self, voc_size, embed_size, num_class):
    super().__init__()
    # voc_size：詞彙表大小 embed_size：嵌入向量的大小 sparse=True：gradient w.r.t.(關於) weight matrix will be a sparse tensor.用以加快訓練速度
    self.embedding = nn.EmbeddingBag(voc_size, embed_size, sparse=False)
    self.rnn = nn.RNN(embed_size, 32)
    self.fc = nn.Linear(32, num_class)  # 輸入張量大小：32，輸出張量大小：num_class
    self.init_weights()

  def init_weights(self):
    init_range = 0.5
    self.embedding.weight.data.uniform_(-init_range, init_range)  # 對 embedding 的權重張量進行均勻分布的隨機初始化
    self.fc.weight.data.uniform_(-init_range, init_range)  # 對 fc 的權重張量進行均勻分布的隨機初始化
    self.fc.bias.data.zero_()  # 將 fc 的 bias 初始化成 0

  def forward(self, text, offsets):  # offsets：張量的起始位置，透過此將這些序列進行累加求和或平均操作，例如：[0, 2]，將 index 0、1 的詞向量相加，2 到最後一個 index 的詞向量相加
    embed = self.embedding(text, offsets)
    rnn_out, h_out = self.rnn(embed)  # rnn 的預設初始隱藏狀態為 0 張量，h_out 為最終隱藏狀態
    fc_out = self.fc(rnn_out)  # run_out.size：[字串長度(時間步數), batch, input_size：詞向量的size)]
    return fc_out

model = TextClassfication(voc_size, embed_size, num_class).to(device)


## 訓練函數
def train(dataloader):
  model.train()
  correct = 0
  total = 0
  interval = 500 # 每 500 個 batch 紀錄一次

  for idx, (label, text, offsets) in enumerate(dataloader):
    optimizer.zero_grad()
    pred = model(text, offsets)
    loss = criterion(pred, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 梯度裁減，用來解決梯度爆炸的問題，max_norm 設定成 0.1
    optimizer.step()
    correct += (pred.argmax(dim=1) == label).sum().item()
    total += label.size(0)  # 加 label 第 0 維度的大小(筆數)
    if idx % interval == 0 and idx > 0:
      print(f'\| epoch {epoch} \| {idx:5d}/{len(dataloader)} batches \| accracy {correct/total:8.3f}')
      correct = 0
      total = 0


## 評估函數
def evaluate(dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            pred = model(text, offsets)
            loss = criterion(pred, label)
            correct += (pred.argmax(dim=1) == label).sum().item()
            total += label.size(0)
    return correct/total


## 建立 dataloader，一批一批訓練
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

def collate_batch(batch):  # 將句子的單字轉成 index 後串起來
    label_list, text_list, offsets = [], [], [0]
    for (label, text) in batch:
        label_list.append(label_trans(label))
        # 在分類任務中，標籤通常用 torch.int64 表示，因为大多數損失函數（例如 nn.CrossEntropyLoss）需要 int64 類型的標籤。
        processed_text = torch.tensor(text_trans(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0)) # 利用句子長度計算每個句子開頭的位置
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # offsets[:-1] 最後一個不用
    text_list = torch.cat(text_list, dim=0)
    return label_list.to(device), text_list.to(device), offsets.to(device)

# 轉換為 DataSet
train_dataset = to_map_style_dataset(train_ds)
test_dataset = to_map_style_dataset(test_ds)

# 將資料切割，95% 做為訓練資料
num_train = int(len(train_dataset) * 0.95)
split_train, split_valid = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

# 建立 DataLoader
train_dataloader = DataLoader(split_train, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)


## 開始訓練
import time
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)  # 用來動態調整 lr，每個 epoch 後 lr 乘以 0.1

accuracy = None
for epoch in range(1, epoches+1):
    start_time = time.time()
    train(train_dataloader)
    val_accuracy = evaluate(valid_dataloader)
    if accuracy is not None and accuracy > val_accuracy:
        scheduler.step()  # 會按照設定的規則更新優化器中的學習率。
    else:
        accuracy = val_accuracy
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch, time.time()-start_time, val_accuracy))
    print('-' * 59)


## 模型對於測試資料的準確度
print(f'測試資料的準確度：{evaluate(test_dataloader):.3f}')


## 預測新資料
ag_news_label = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tec"}

# 預測函數
def predict(text, text_trans):
    with torch.no_grad():
        text = torch.tensor(text_trans(text)).to(device)
        print(text)
        output = model(text, torch.tensor([0])).to(device)
        return output.argmax(dim=1).item() + 1

# 测试资料
ex_text_str = "MEMPHIS, Tenn. – Four days ago, Jon Rahm was \
    enduring the season’s worst weather conditions on Sunday at The \
    Open on his way to a closing 75 at Royal Portrush, which \
    considering the wind and the rain was a respectable showing. \
    Thursday’s first round at the WGC-FedEx St. Jude Invitational \
    was another story. With temperatures in the mid-80s and hardly any \
    wind, the Spaniard was 13 strokes better in a flawless round. \
    Thanks to his best putting performance on the PGA Tour, Rahm \
    finished with an 8-under 62 for a three-stroke lead, which \
    was even more impressive considering he’d never played the \
    front nine at TPC Southwind."

print(ag_news_label[predict(ex_text_str, text_trans)])