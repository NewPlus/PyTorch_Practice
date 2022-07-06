import torch
import torch.nn as nn

# 배치 크기 × 채널 × 높이(height) × 너비(widht)의 크기의 텐서를 선언
inputs = torch.Tensor(1, 1, 28, 28)
print('텐서의 크기 : {}'.format(inputs.shape))

# 채널 = 1, 출력 채널 = 32, 커널 사이즈 = 3, 패딩 = 1
conv1 = nn.Conv2d(1, 32, 3, padding=1)
print(conv1)

# 채널 = 32, 출력 채널 = 64, 커널 사이즈 = 3, 패딩 = 1
conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
print(conv2)

# 스트라이드 = 2, 커널 사이즈 = 2
pool = nn.MaxPool2d(2)
print(pool)

# CNN의 출력 채널이 32이므로 텐서의 채널 수는 32채널
# 28 x 28은 패딩이 1이므로 크기가 보존되었기 때문
out = conv1(inputs)
print(out.shape)

# 스트라이드와 커널 사이즈가 2인 맥스풀링에서
# 28 x 28 크기가 14 x 14로 바뀜
out = pool(out)
print(out.shape)

# CNN의 출력 채널이 64이므로 텐서의 채널 수는 64채널
# 14 x 14은 패딩이 1이므로 크기가 보존되었기 때문
out = conv2(out)
print(out.shape)

# 스트라이드와 커널 사이즈가 2인 맥스풀링에서
# 14 x 14 크기가 7 x 7로 바뀜
out = pool(out)
print(out.shape)

# 첫번째 차원인 배치 차원은 그대로 두고 나머지는 펼쳐라
# 1 x 64 x 7 x 7 = 3136
out = out.view(out.size(0), -1) 
print(out.shape)

# 3136의 input_data를 10의 out_data로 변환
fc = nn.Linear(3136, 10) # input_dim = 3,136, output_dim = 10
out = fc(out)
print(out.shape)