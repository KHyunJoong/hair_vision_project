# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# # 더블 컨볼루션: 두 번의 컨볼루션 연산 (컨볼루션 -> 배치 정규화 -> ReLU)
# class DoubleConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(DoubleConv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         return self.conv(x)
#
# # 다운샘플링 (MaxPooling 후 DoubleConv)
# class Down(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Down, self).__init__()
#         self.maxpool_conv = nn.Sequential(
#             nn.MaxPool2d(2),
#             DoubleConv(in_channels, out_channels)
#         )
#
#     def forward(self, x):
#         return self.maxpool_conv(x)
#
# # 업샘플링 (업샘플링 후 이전 계층과의 concat 그리고 DoubleConv)
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels, bilinear=True):
#         super(Up, self).__init__()
#         # bilinear upsampling 혹은 transposed convolution 선택
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
#         self.conv = DoubleConv(in_channels, out_channels)
#
#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         # x1과 x2의 공간 크기가 다를 수 있으므로 패딩으로 맞춤
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]
#         x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
#                         diffY // 2, diffY - diffY // 2])
#         # 채널 차원에서 concatenate
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)
#
# # 출력 레이어: 1x1 컨볼루션을 통해 원하는 클래스 수로 매핑
# class OutConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(OutConv, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         return self.conv(x)
#
# # U-Net 전체 아키텍처 구현
# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         factor = 2 if bilinear else 1
#         self.down4 = Down(512, 1024 // factor)
#         self.up1 = Up(1024, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)
#
#     def forward(self, x):
#         x1 = self.inc(x)      # encoder의 첫 번째 블록
#         x2 = self.down1(x1)   # 두 번째 블록
#         x3 = self.down2(x2)   # 세 번째 블록
#         x4 = self.down3(x3)   # 네 번째 블록
#         x5 = self.down4(x4)   # bottleneck
#         x = self.up1(x5, x4)  # decoder 첫 번째 업샘플링 (skip connection 사용)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits
#
# # 모델 생성 및 테스트 예시
# if __name__ == '__main__':
#     # 예시: 3채널 이미지, 1개의 출력 채널(예: binary segmentation)
#     model = UNet(n_channels=3, n_classes=1)
#     print(model)
#
#     # 더미 데이터 생성 후 모델 통과 테스트 (배치 크기: 1, 크기: 256x256)
#     x = torch.randn(1, 3, 256, 256)
#     output = model(x)
#     print("Output shape:", output.shape)
#
# ##################################################
# import cv2
# import torch
# import numpy as np
# from torchvision import transforms
#
# # 위에서 구현한 UNet 클래스를 포함했다고 가정합니다.
# # model = UNet(n_channels=3, n_classes=1) 를 이용해 모델 생성
# model = UNet(n_channels=3, n_classes=1)
# model.eval()  # 평가 모드로 전환
#
# # 이미지 로드 및 전처리 함수
# def preprocess_image(image_path, target_size=(256, 256)):
#     # 이미지 읽기 (cv2는 기본적으로 BGR)
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("이미지 파일을 찾을 수 없습니다: " + image_path)
#     # BGR -> RGB 변환
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # 크기 조정
#     image = cv2.resize(image, target_size)
#     # Tensor로 변환하고 정규화 (예시로 ImageNet 통계 사용)
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ])
#     image_tensor = transform(image)
#     # 배치 차원 추가: (1, C, H, W)
#     image_tensor = image_tensor.unsqueeze(0)
#     return image_tensor, image
#
# # 이미지 전처리
# image_path = "testcut.jpg"  # 확인할 이미지 파일 경로
# try:
#     input_tensor, original_image = preprocess_image(image_path)
# except ValueError as e:
#     print(e)
#     exit()
#
# # 모델에 이미지 전달하여 예측
# with torch.no_grad():
#     output = model(input_tensor)
#
# # 예측 결과 처리 (binary segmentation인 경우 sigmoid 후 thresholding)
# output = torch.sigmoid(output)
# # 배치 차원 제거 및 numpy array 변환
# output_np = output.squeeze().cpu().numpy()
#
# # 예시로 threshold 0.5를 기준으로 이진화
# segmentation = (output_np > 0.5).astype(np.uint8) * 255
#
# # 결과 시각화: 원본 이미지와 분할 결과 출력 (cv2.imshow는 BGR 형식을 기대하므로, 원본 이미지는 다시 BGR로 변환)
# original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
# cv2.imshow("Original Image", original_bgr)
# cv2.imshow("Segmentation", segmentation)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from os import listdir
from os.path import join
# from train_cvppp import evaluate

import deepcoloring as dc

import matplotlib.pyplot as plt

# 추가: plt.show() 명시적으로 호출
plt.show()

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# !wget https://www.dropbox.com/s/whwzh9pdbk69o7x/cvppp_model.t7 -O model.t7

net = dc.EUnet(3, 9, 4, 3, 1, depth=3, padding=1, init_xavier=True, use_bn=False, use_dropout=True).to(device)

net.load_state_dict(torch.load("cvppp_model.t7", weights_only=False))
net.eval()
print("Model loaded")


from skimage.io import imread
xo = imread("5e95fbe7a58a6ee7d6027273facf7bc6.jpg")[::2,::2]

x = dc.rgba2rgb()(xo, True)/255.
x = dc.normalize(0.5, 0.5, )(x, True)
x = x.transpose(2, 0, 1)[:, :248, :248]

vx = torch.from_numpy(np.expand_dims(x, 0)).to(device)
p = net(vx)
p_numpy = p.detach().cpu().numpy()[0]
dc.visualize(xo[:,:,:3],p_numpy,65)
plt.show()  # 강제 표시
