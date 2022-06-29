import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더

# 데이터
x_train = torch.FloatTensor([[78, 80, 98, 75],
                            [84, 76, 87, 72],
                            [79, 87, 95, 80],
                            [96, 92, 94, 100],
                            [75, 92, 85, 89]])
y_train = torch.FloatTensor([[142], [138], [171], [152], [169]])

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(4,1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

nb_epochs = 1000000
for epoch in range(nb_epochs + 1):
  for batch_idx, samples in enumerate(dataloader):
    # print(batch_idx)
    # print(samples)
    x_train, y_train = samples
    # H(x) 계산
    prediction = model(x_train)

    # cost 계산
    cost = F.mse_loss(prediction, y_train)

    # cost로 H(x) 계산
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100000 == 0:
        # 100000번마다 로그 출력
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx+1, len(dataloader),
            cost.item()
            ))

# 임의의 입력 [78, 80, 98, 75]를 선언
new_var =  torch.FloatTensor([[78, 80, 98, 75]]) 
# 입력한 값 [78, 80, 98, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 78, 80, 98, 75일 때의 예측값 :", pred_y)
