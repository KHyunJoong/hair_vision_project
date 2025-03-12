
import os
import csv
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        pool = self.pool(skip)
        return skip, pool

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(input_channels, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.bridge = ConvBlock(512, 1024)

        self.decoder1 = DecoderBlock(1024, 512)
        self.decoder2 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder4 = DecoderBlock(128, 64)

        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.encoder1(x)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        b = self.bridge(p4)

        d1 = self.decoder1(b, s4)
        d2 = self.decoder2(d1, s3)
        d3 = self.decoder3(d2, s2)
        d4 = self.decoder4(d3, s1)

        outputs = torch.sigmoid(self.final_conv(d4))
        return outputs
def load_model(model, epochs,result_dir):
    path=f"{result_dir}/unet_model_{epochs}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"✅ 모델이 불러와졌습니다: {path}")

def load_data2(images_dir, image_size=512, test_size=0.2, val_size=0.2):
    """
    이미지 및 마스크 데이터를 로드하고, Train/Validation/Test 데이터로 분할하는 함수

    Args:
        images_dir (str): 이미지 폴더 경로
        masks_dir (str): 마스크 폴더 경로
        image_size (int, optional): 이미지 크기 (기본값: 512)
        test_size (float, optional): Test 데이터 비율 (기본값: 0.2)
        val_size (float, optional): Train 데이터에서 Validation 데이터로 사용할 비율 (기본값: 0.2)

    Returns:
        Tuple: (images_train, images_val, images_test, masks_train, masks_val, masks_test)
    """
    images_listdir = sorted(os.listdir(images_dir))

    print(f"총 이미지 개수: {len(images_listdir)}")

    # 이미지 및 마스크 데이터 로드
    images = []
    for j, file in enumerate(images_listdir):
        try:
            image = cv2.imread(os.path.join(images_dir, file))
            image = cv2.resize(image, (image_size, image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
            images.append(image)

        except Exception as e:
            print(f"파일 로드 오류: {file} - {e}")
            continue

    images = np.array(images, dtype=np.uint8)

    print(f"로드된 이미지 형태: {images.shape}")

    print(f"Test: {images.shape}")
    return images

def save_result_image2(idx, og, unet, p, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(20, 12))


    # ✅ 원본 이미지 → 2번 위치
    axs[0].imshow(og)
    axs[0].set_title(f"Original {idx}")
    axs[0].axis("off")

    # ✅ 예측 마스크 → 3번 위치
    axs[1].imshow(unet, cmap="gray")
    axs[1].set_title(f"U-Net Prediction (p > {p})")
    axs[1].axis("off")

    # ✅ 이미지 저장 (IoU & Dice Score 포함)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

def test_data_load(images_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_test_torch = torch.tensor(images_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    images_test_torch = images_test_torch.to(device)
    return images_test_torch

def model_test2(test_data, images_test, epochs, Threshold, model_dir,result_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=3, output_channels=1).to(device)
    load_model(model, epochs, model_dir)

    # 모델을 평가 모드로 전환
    model.eval()

    # 모델 예측
    with torch.no_grad():
        unet_predict = model(test_data)
        unet_predict = torch.sigmoid(unet_predict)  # BCE Loss 사용 시 필요

    # NumPy 변환
    unet_predict = unet_predict.cpu().numpy().squeeze(1)  # (batch, 1, H, W) → (batch, H, W)

    # Threshold 적용
    r_values = Threshold
    unet_predictions = [(unet_predict > r).astype(np.uint8) for r in r_values]

    # CSV 저장 파일 생성
    csv_file = os.path.join(result_dir, "test_results.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "Threshold", "IoU", "Dice Score"])

        for idx in range(len(unet_predict)):
            for i, r in enumerate(r_values):

                writer.writerow([idx, r])
                above_threshold = np.sum(unet_predictions[i][idx] > 0)  # 1로 변환된 픽셀 개수
                below_threshold = np.sum(unet_predictions[i][idx] == 0) # 0인 픽셀 개수


                save_path = os.path.join(result_dir, f"result_{idx}_threshold_{r}.png")
                save_result_image2(idx, images_test[idx], unet_predictions[i][idx], r, save_path)
                print(f"[Index {idx}, Threshold {r}]")
                print(f"[Index {idx}, Threshold {r}] Above: {above_threshold}, Below: {below_threshold}")
    np.save(os.path.join(result_dir, "unet_predictions.npy"), unet_predict)
    del model
    torch.cuda.empty_cache()
    print(f"✅ 테스트 결과 저장 완료! 폴더: {result_dir}")
    return above_threshold + below_threshold, above_threshold
images_dir = "./datasets/hair/images/validation3"
model_dir = "./unet/test_results2025-03-08_10-22-17"
result_dir = "./unet/test3"
epochs = 200
Threshold = [0.65]
image_size = 512
# ✅ 데이터 로드
images_test= load_data2(images_dir, image_size)
model_test2(test_data_load(images_test), images_test, epochs,Threshold, model_dir, result_dir)