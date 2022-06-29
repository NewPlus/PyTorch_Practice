import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)

# 훈련 데이터
x_train = torch.FloatTensor([[78, 80, 98, 75],
                            [84, 76, 87, 72],
                            [79, 87, 95, 80],
                            [96, 92, 94, 100],
                            [75, 92, 85, 89]])
y_train = torch.FloatTensor([[142], [138], [171], [152], [169]])

# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=4, output_dim=1.
model = nn.Linear(4,1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

nb_epochs = 1000000
for epoch in range(nb_epochs+1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train) # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 10000 == 0:
    # 10000번마다 로그 출력
      print('Epoch {:4d}/{} Cost: {:.6f}'.format(
          epoch, nb_epochs, cost.item()
      ))

# 임의의 입력 [78, 80, 98, 75]를 선언
new_var =  torch.FloatTensor([[78, 80, 98, 75]]) 
# 입력한 값 [78, 80, 98, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var) 
print("훈련 후 입력이 78, 80, 98, 75일 때의 예측값 :", pred_y) 
print(list(model.parameters()))