import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
import time
import datetime
from torch.utils.data import DataLoader, TensorDataset
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
#이미지 위치변경 및 파일 세팅
#
# paths=[]
# for dirname, _, filenames in os.walk('./testdata'):
#     for filename in filenames:
#         paths+=[(os.path.join(dirname, filename))]
# print(paths)
# os.makedirs("./testdata/images", exist_ok=True)  # 이미 존재하면 무시
# os.makedirs("./testdata/masks", exist_ok=True)
#
# for path in paths:
#     file = os.path.basename(path)  # 파일 이름만 추출
#     if file.startswith("mask"):
#         shutil.move(path, "./testdata/masks")  # 파일을 masks 폴더로 이동 (잘라내기)
#     elif file.endswith(".png"):
#         shutil.move(path, "./testdata/images")  # 파일을 images 폴더로 이동 (잘라내기)

#수정중
# images_dir ='./testdata/images'
# masks_dir = './testdata/masks'
#
# images_listdir = sorted(os.listdir(images_dir))
# masks_listdir = sorted(os.listdir(masks_dir))
# N=list(range(9))
# random_N = N
#
# print(len(images_listdir))
# print(len(masks_listdir))
#
# image_size=512
# input_image_size=(512,512)
#
# def read_image(path):
#     img = cv2.imread(path)
#     img = cv2.resize(img, (image_size, image_size))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img
#
# number=120
#
# MASKS=np.zeros((1,image_size, image_size, 1), dtype=bool)
# IMAGES=np.zeros((1,image_size, image_size, 3),dtype=np.uint8)
#
# for j,file in enumerate(images_listdir):   ##the smaller, the faster
#     try:
#         image = read_image(f"{images_dir}/{file}")
#         image_ex = np.expand_dims(image, axis=0)
#         IMAGES = np.vstack([IMAGES, image_ex])
#         mask = read_image(f"{masks_dir}/{masks_listdir[j]}")
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#         mask = mask.reshape(512,512,1)
#         mask_ex = np.expand_dims(mask, axis=0)
#         MASKS = np.vstack([MASKS, mask_ex])
#     except:
#         print(file)
#         continue
#
# images=np.array(IMAGES)[1:number+1]
# masks=np.array(MASKS)[1:number+1]
# print(images.shape,masks.shape)
#
# images_train, images_test, masks_train, masks_test = train_test_split(
#     images, masks, test_size=0.2, random_state=42)


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
def save_model(model,epochs,result_dir ):
    path=f"{result_dir}/unet_model_{epochs}.pth"
    torch.save(model.state_dict(), path)
    print(f"✅ 모델이 저장되었습니다: {path}")
def load_model(model, epochs,result_dir):
    path=f"{result_dir}/unet_model_{epochs}.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"✅ 모델이 불러와졌습니다: {path}")

class UNet_multi(nn.Module):
    def __init__(self, input_channels=3, output_channels=3):
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

        outputs = torch.softmax(self.final_conv(d4), dim=1)  # ✅ softmax 적용
        return outputs

# 모델 생성 및 테스트
# model = UNet(input_channels=3, output_channels=1)
# print(model)
def load_data(images_dir, masks_dir, image_size=512, test_size=0.2, val_size=0.2):
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
    masks_listdir = sorted(os.listdir(masks_dir))

    print(f"총 이미지 개수: {len(images_listdir)}")
    print(f"총 마스크 개수: {len(masks_listdir)}")

    # 이미지 및 마스크 데이터 로드
    images, masks = [], []
    for j, file in enumerate(images_listdir):
        try:
            image = cv2.imread(os.path.join(images_dir, file))
            image = cv2.resize(image, (image_size, image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
            images.append(image)

            mask = cv2.imread(os.path.join(masks_dir, masks_listdir[j]), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (image_size, image_size))
            mask = np.expand_dims(mask, axis=-1)  # (H, W) → (H, W, 1)
            mask = (mask > 127).astype(np.uint8)  # Threshold 적용하여 0과 1로 변환
            masks.append(mask)

        except Exception as e:
            print(f"파일 로드 오류: {file} - {e}")
            continue

    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)

    print(f"로드된 이미지 형태: {images.shape}, 마스크 형태: {masks.shape}")

    # ✅ Train/Test Split (Test 데이터 먼저 분리)
    images_train_full, images_test, masks_train_full, masks_test = train_test_split(
        images, masks, test_size=test_size, random_state=42
    )

    # ✅ Train/Validation Split (Train 데이터에서 Validation 분리)
    images_train, images_val, masks_train, masks_val = train_test_split(
        images_train_full, masks_train_full, test_size=val_size, random_state=42
    )

    print(f"Train: {images_train_full.shape}, Validation: {images_val.shape}, Test: {images_test.shape}")
    return images_train_full, images_val, images_test, masks_train_full, masks_val, masks_test
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_multi(images_dir, masks_dir, image_size=512, test_size=0.2, val_size=0.2):
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
    masks_listdir = sorted(os.listdir(masks_dir))

    print(f"총 이미지 개수: {len(images_listdir)}")
    print(f"총 마스크 개수: {len(masks_listdir)}")

    # 이미지 및 마스크 데이터 로드
    images, masks = [], []
    for j, file in enumerate(images_listdir):
        try:
            # 이미지 로드
            image = cv2.imread(os.path.join(images_dir, file))
            image = cv2.resize(image, (image_size, image_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
            images.append(image)

            # 마스크 로드 (Grayscale)
            mask = cv2.imread(os.path.join(masks_dir, masks_listdir[j]), cv2.IMREAD_GRAYSCALE)
            print("mask",np.unique(mask))
            mask = cv2.resize(mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)
            print("mask",np.unique(mask))
            # ✅ Multi-Class Mask 변환 (0, 1, 2)
            mask[mask == 255] = 2  # 흰색 → 클래스 2
            mask[mask == 128] = 1  # 회색 → 클래스 1
            mask[mask == 0] = 0    # 검은색 → 클래스 0

            # (H, W) → (H, W, 1)로 차원 확장
            mask = np.expand_dims(mask, axis=-1)
            masks.append(mask)

        except Exception as e:
            print(f"파일 로드 오류: {file} - {e}")
            continue

    images = np.array(images, dtype=np.uint8)
    masks = np.array(masks, dtype=np.uint8)

    print(f"로드된 이미지 형태: {images.shape}, 마스크 형태: {masks.shape}")

    # ✅ Train/Test Split (Test 데이터 먼저 분리)
    images_train_full, images_test, masks_train_full, masks_test = train_test_split(
        images, masks, test_size=test_size, random_state=42
    )

    # ✅ Train/Validation Split (Train 데이터에서 Validation 분리)
    images_train, images_val, masks_train, masks_val = train_test_split(
        images_train_full, masks_train_full, test_size=val_size, random_state=42
    )

    print(f"Train: {images_train.shape}, Validation: {images_val.shape}, Test: {images_test.shape}")
    return images_train, images_val, images_test, masks_train, masks_val, masks_test

# ✅ Augmentation 설정 함수 (켜고 끌 수 있도록 설정)
def get_augmentation(apply_augmentation=True):
    """
    Augmentation을 적용할지 여부에 따라 변환을 반환하는 함수

    Args:
        apply_augmentation (bool): Augmentation 적용 여부 (True: 적용, False: 미적용)

    Returns:
        albumentations.Compose: Augmentation 변환 객체
    """
    if apply_augmentation:
        return A.Compose([
            A.HorizontalFlip(p=0.5),  # 좌우 반전 (50% 확률)
            A.VerticalFlip(p=0.3),  # 상하 반전 (30% 확률)
            A.RandomRotate90(p=0.5),  # 90도 회전 (50% 확률)
            A.RandomBrightnessContrast(p=0.2),  # 밝기 & 대비 조정 (20% 확률)
            A.GaussianBlur(p=0.2),  # 가우시안 블러 (20% 확률)
            A.ElasticTransform(p=0.2, alpha=1, sigma=50, approximate=True),  # 탄성 변형 (20% 확률)
            A.Resize(512, 512),  # 크기 조정 (모든 이미지 512x512로 통일)
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),  # 크기 조정만 적용 (Augmentation 미적용)
        ])

# ✅ Augmentation을 적용하는 함수 (True / False로 설정)
def apply_augmentation(images, masks, apply_aug=True, augmentation_factor=10):
    """
    Augmentation 적용 여부에 따라 데이터를 변환하는 함수

    Args:
        images (numpy.ndarray): 원본 이미지 배열 (H, W, C)
        masks (numpy.ndarray): 원본 마스크 배열 (H, W, 1)
        apply_aug (bool): Augmentation 적용 여부 (True: 적용, False: 미적용)

    Returns:
        numpy.ndarray: Augmentation이 적용된 이미지 및 마스크
    """
    transform = get_augmentation(apply_aug)
    augmented_images, augmented_masks = [], []
    if apply_aug:
        print('aug_T')
        for img, mask in zip(images, masks):
            augmented_images.append(img)  # 원본 추가
            augmented_masks.append(mask)

            for _ in range(augmentation_factor - 1):  # ✅ N배 증강
                augmented = transform(image=img, mask=mask)
                augmented_images.append(augmented['image'])
                augmented_masks.append(augmented['mask'])

        return np.array(augmented_images), np.array(augmented_masks)
    else:
        print('aug_F')
        return  images, masks


# ✅ Augmentation 적용 후 시각화 함수
def visualize_augmentation(images, masks, apply_aug=True, num_samples=3):
    fig, axs = plt.subplots(num_samples, 4, figsize=(15, 5 * num_samples))

    # ✅ Augmentation 적용 여부 선택 (함수와 충돌 방지)
    images_aug, masks_aug = apply_augmentation(images, masks, apply_aug)

    for i in range(num_samples):
        idx = random.randint(0, len(images) - 1)

        # 원본 및 증강된 이미지 및 마스크 가져오기
        original_image = images[idx]
        original_mask = masks[idx]
        transformed_image = images_aug[idx]
        transformed_mask = masks_aug[idx]

        axs[i, 0].imshow(original_image)
        axs[i, 0].set_title("Original Image")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(transformed_image)
        axs[i, 1].set_title("Augmented Image" if apply_augmentation else "Resized Image")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(original_mask, cmap="gray")
        axs[i, 2].set_title("Original Mask")
        axs[i, 2].axis("off")

        axs[i, 3].imshow(transformed_mask, cmap="gray")
        axs[i, 3].set_title("Augmented Mask" if apply_augmentation else "Resized Mask")
        axs[i, 3].axis("off")

    plt.show()

# ✅ Train DataLoader 생성 함수
# ✅ Train DataLoader 생성 함수 (Augmentation On/Off 가능)
# def train_data_load(images_train, masks_train, batch_size, apply_aug=False, augmentation_factor=10):
#     """
#     Train 데이터를 Augmentation 후 PyTorch DataLoader로 변환하는 함수
#
#     Args:
#         images_train (numpy.ndarray): 학습 이미지 (HWC 형식)
#         masks_train (numpy.ndarray): 학습 마스크 (HWC 형식)
#         batch_size (int): 배치 크기
#         apply_aug (bool): Augmentation 적용 여부 (True: 적용, False: 미적용)
#         augmentation_factor (int): Augmentation 적용 시 몇 배로 증강할지 설정
#
#     Returns:
#         DataLoader: PyTorch DataLoader 객체
#     """
#     # ✅ 증강 전 데이터 크기 확인
#     print(f"🔹 Augmentation 전 데이터 크기: images = {images_train.shape}, masks = {masks_train.shape}")
#
#     # ✅ Train DataLoader 생성 함수 (Augmentation On/Off 가능)
#     images_train_aug, masks_train_aug = apply_augmentation(images_train, masks_train, apply_aug, augmentation_factor)
#
#     # ✅ 증강 후 데이터 크기 확인
#     print(f"✅ Augmentation 적용 후 데이터 크기: images = {images_train_aug.shape}, masks = {masks_train_aug.shape}")
#
#     # PyTorch Tensor 변환
#     images_train_torch = torch.tensor(images_train_aug, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
#     masks_train_torch = torch.tensor(masks_train_aug, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # 정규화
#
#     train_dataset = TensorDataset(images_train_torch, masks_train_torch)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#
#     return train_loader


def train_data_load(images_train, masks_train, batch_size, apply_aug=False, augmentation_factor=10):
    """
    Train 데이터를 Augmentation 후 PyTorch DataLoader로 변환하는 함수

    Args:
        images_train (numpy.ndarray): 학습 이미지 (HWC 형식)
        masks_train (numpy.ndarray): 학습 마스크 (HWC 형식)
        batch_size (int): 배치 크기
        apply_aug (bool): Augmentation 적용 여부 (True: 적용, False: 미적용)
        augmentation_factor (int): Augmentation 적용 시 몇 배로 증강할지 설정

    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    # ✅ 증강 전 데이터 크기 확인
    print(f"🔹 Augmentation 전 데이터 크기: images = {images_train.shape}, masks = {masks_train.shape}")

    # ✅ 마스크 값 범위 확인 (정규화 필요 여부 확인)
    print(f"🔍 마스크 데이터 최소값: {masks_train.min()}, 최대값: {masks_train.max()}")

    # ✅ Train DataLoader 생성 함수 (Augmentation On/Off 가능)
    images_train_aug, masks_train_aug = apply_augmentation(images_train, masks_train, apply_aug, augmentation_factor)

    # ✅ 증강 후 데이터 크기 확인
    print(f"✅ Augmentation 적용 후 데이터 크기: images = {images_train_aug.shape}, masks = {masks_train_aug.shape}")

    # ✅ Augmentation 후 마스크 값 범위 확인 (반전 여부 체크)
    print(f"🔍 증강 후 마스크 데이터 최소값: {masks_train_aug.min()}, 최대값: {masks_train_aug.max()}")

    # PyTorch Tensor 변환
    images_train_torch = torch.tensor(images_train_aug, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    masks_train_torch = torch.tensor(masks_train_aug, dtype=torch.float32).permute(0, 3, 1, 2)   # 정규화

    # ✅ PyTorch Tensor 변환 후 마스크 값 범위 확인
    print(f"🛠 변환 후 마스크 최소값: {masks_train_torch.min()}, 최대값: {masks_train_torch.max()}")

    train_dataset = TensorDataset(images_train_torch, masks_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader



# ✅ Validation DataLoader 생성 함수 (Augmentation On/Off 가능)
# ✅ Validation DataLoader 생성 함수 (Augmentation On/Off 가능)
def val_data_load(images_val, masks_val, batch_size, apply_aug=False, augmentation_factor=1):
    """
    Validation 데이터를 Augmentation 후 PyTorch DataLoader로 변환하는 함수

    Args:
        images_val (numpy.ndarray): 검증 이미지 (HWC 형식)
        masks_val (numpy.ndarray): 검증 마스크 (HWC 형식)
        batch_size (int): 배치 크기
        apply_aug (bool): Augmentation 적용 여부 (True: 적용, False: 미적용)
        augmentation_factor (int): 증강 횟수 (Validation에는 기본값 1)

    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    # Augmentation 적용 여부 선택
    images_val_aug, masks_val_aug = apply_augmentation(images_val, masks_val, apply_aug, augmentation_factor)

    # PyTorch Tensor 변환
    images_val_torch = torch.tensor(images_val_aug, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    masks_val_torch = torch.tensor(masks_val_aug, dtype=torch.float32).permute(0, 3, 1, 2)   # 정규화

    print("Validation Masks min/max:", masks_val_torch.min().item(), masks_val_torch.max().item())

    val_dataset = TensorDataset(images_val_torch, masks_val_torch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 검증 데이터는 일반적으로 shuffle=False
    return val_loader

def train_data_load_multi(images_train, masks_train, batch_size, apply_aug=False, augmentation_factor=10):
    """
    Train 데이터를 Augmentation 후 PyTorch DataLoader로 변환하는 함수

    Args:
        images_train (numpy.ndarray): 학습 이미지 (HWC 형식)
        masks_train (numpy.ndarray): 학습 마스크 (HWC 형식)
        batch_size (int): 배치 크기
        apply_aug (bool): Augmentation 적용 여부 (True: 적용, False: 미적용)
        augmentation_factor (int): Augmentation 적용 시 몇 배로 증강할지 설정

    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    # ✅ 증강 전 데이터 크기 확인
    print(f"🔹 Augmentation 전 데이터 크기: images = {images_train.shape}, masks = {masks_train.shape}")

    # ✅ 마스크 값 범위 확인 (정규화 필요 여부 확인)
    print(f"🔍 마스크 데이터 최소값: {masks_train.min()}, 최대값: {masks_train.max()}")

    # ✅ Train DataLoader 생성 함수 (Augmentation On/Off 가능)
    images_train_aug, masks_train_aug = apply_augmentation(images_train, masks_train, apply_aug, augmentation_factor)

    # ✅ 증강 후 데이터 크기 확인
    print(f"✅ Augmentation 적용 후 데이터 크기: images = {images_train_aug.shape}, masks = {masks_train_aug.shape}")

    # ✅ Augmentation 후 마스크 값 범위 확인 (반전 여부 체크)
    print(f"🔍 증강 후 마스크 데이터 최소값: {masks_train_aug.min()}, 최대값: {masks_train_aug.max()}")

    # PyTorch Tensor 변환
    images_train_torch = torch.tensor(images_train_aug, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    masks_train_torch = torch.tensor(masks_train_aug, dtype=torch.float32).permute(0, 3, 1, 2) /masks_train_aug.max() # 정규화

    # ✅ PyTorch Tensor 변환 후 마스크 값 범위 확인
    print(f"🛠 변환 후 마스크 최소값: {masks_train_torch.min()}, 최대값: {masks_train_torch.max()}")

    train_dataset = TensorDataset(images_train_torch, masks_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader



# ✅ Validation DataLoader 생성 함수 (Augmentation On/Off 가능)
# ✅ Validation DataLoader 생성 함수 (Augmentation On/Off 가능)
def val_data_load_multi(images_val, masks_val, batch_size, apply_aug=False, augmentation_factor=1):
    """
    Validation 데이터를 Augmentation 후 PyTorch DataLoader로 변환하는 함수

    Args:
        images_val (numpy.ndarray): 검증 이미지 (HWC 형식)
        masks_val (numpy.ndarray): 검증 마스크 (HWC 형식)
        batch_size (int): 배치 크기
        apply_aug (bool): Augmentation 적용 여부 (True: 적용, False: 미적용)
        augmentation_factor (int): 증강 횟수 (Validation에는 기본값 1)

    Returns:
        DataLoader: PyTorch DataLoader 객체
    """
    # Augmentation 적용 여부 선택
    images_val_aug, masks_val_aug = apply_augmentation(images_val, masks_val, apply_aug, augmentation_factor)

    # PyTorch Tensor 변환
    images_val_torch = torch.tensor(images_val_aug, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    masks_val_torch = torch.tensor(masks_val_aug, dtype=torch.float32).permute(0, 3, 1, 2) /masks_val_aug.max()  # 정규화

    print("Validation Masks min/max:", masks_val_torch.min().item(), masks_val_torch.max().item())

    val_dataset = TensorDataset(images_val_torch, masks_val_torch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # 검증 데이터는 일반적으로 shuffle=False
    return val_loader
# # 데이터 변환 (0~255 값을 0~1로 정규화)
# def train_data_load(images_train, masks_train, batch_size):
#     images_train_torch = torch.tensor(images_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
#     masks_train_torch = torch.tensor(masks_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # ✅ 마스크도 정규화
#
#     # 마스크 값이 0~1 사이인지 확인
#     print("Masks min/max:", masks_train_torch.min().item(), masks_train_torch.max().item())
#
#     # PyTorch DataLoader 설정
#     train_dataset = TensorDataset(images_train_torch, masks_train_torch)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     return train_loader
# def val_data_load(images_train, masks_train, batch_size):
#     images_val_torch = torch.tensor(images_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
#     masks_val_torch = torch.tensor(masks_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0  # ✅ 마스크도 정규화
#
#     # 마스크 값이 0~1 사이인지 확인
#     print("Masks min/max:", masks_val_torch.min().item(), masks_val_torch.max().item())
#
#     # PyTorch DataLoader 설정
#     val_dataset = TensorDataset(images_val_torch, masks_val_torch)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
#     return train_loader
def compute_iou(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection / union).item() if union > 0 else 1.0

def compute_dice(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    return (2. * intersection / (preds.sum() + targets.sum())).item() if (preds.sum() + targets.sum()) > 0 else 1.0

def model_train(train_data, val_data, epochs, learning_rate,log_file,result_dir):
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=3, output_channels=1).to(device)
    criterion = nn.BCELoss()  # 바이너리 크로스엔트로피
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # CSV 파일로 로깅
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "IoU", "Dice Score", "Time (s)", "Learning Rate", "GPU Memory (MB)"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.perf_counter()

        for images, masks in train_data:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)  # ✅ masks 값이 0~1 범위여야 함
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_data)

        # ✅ Validation 과정 추가
        model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []

        with torch.no_grad():
            for images, masks in val_data:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                # IoU, Dice Score 계산
                iou = compute_iou(outputs, masks)
                dice = compute_dice(outputs, masks)
                iou_scores.append(iou)
                dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_data)
        avg_iou = np.mean(iou_scores)
        avg_dice = np.mean(dice_scores)
        elapsed_time = time.perf_counter() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB 단위

        # 로그 기록
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_iou, avg_dice, elapsed_time, current_lr, gpu_memory])

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Time: {elapsed_time:.2f}s | LR: {current_lr:.6f} | GPU: {gpu_memory:.2f}MB")

    save_model(model, epochs,result_dir)
    print("Model saved!")
    print("Training complete!")


def model_train_3(train_data, val_data, epochs, learning_rate, log_file, result_dir):
    # Hyperparameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=3, output_channels=3).to(device)  # ✅ 출력 채널 3개로 변경
    criterion = nn.CrossEntropyLoss()  # ✅ 다중 클래스 분류를 위한 손실 함수
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # CSV 파일로 로깅
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "IoU", "Dice Score", "Time (s)", "Learning Rate", "GPU Memory (MB)"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.perf_counter()

        for images, masks in train_data:
            images = images.clone().detach().to(device, dtype=torch.float32)  # ✅ 모델 입력 float32 변환
            masks = masks.clone().detach().to(device, dtype=torch.long)  # ✅ 마스크 정수형 변환

            # 디버깅 출력
            # print("변환 전 masks.shape:", masks.shape)  # (batch, 1, H, W) 또는 (batch, H, W)

            masks = masks.squeeze(1)  # ✅ 다시 한 번 squeeze 적용 (확실히 변환)

            # print("변환 후 masks.shape:", masks.shape)  # (batch, H, W)로 변경되어야 함

            outputs = model(images)  # 모델 실행
            # print("✅ outputs.shape:", outputs.shape)  # (batch, num_classes, H, W) 확인

            loss = criterion(outputs, masks)  # ✅ CrossEntropyLoss 사용
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            # print("masks.shape before squeeze:", masks.shape)
            masks = masks.squeeze(1)  # ✅ 1채널 제거
            # print("masks.shape after squeeze:", masks.shape)

            # 데이터 타입 확인
            # print("✅ masks.dtype:", masks.dtype)  # 반드시 torch.long이어야 함
        avg_train_loss = running_loss / len(train_data)


        # ✅ Validation 과정 추가
        model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []

        with torch.no_grad():
            for images, masks in val_data:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long).squeeze(1)  # ✅ 마스크 차원 줄이기

                # ✅ One-hot Encoding 변환 (CrossEntropyLoss를 사용할 경우 필요할 수도 있음)
                masks = torch.nn.functional.one_hot(masks, num_classes=3).permute(0, 3, 1, 2).float()

                # 디버깅 출력
                # print("✅ outputs.shape:", outputs.shape)  # (batch, 3, H, W)
                # print("✅ masks.shape after one-hot:", masks.shape)  # (batch, 3, H, W)

                outputs = model(images)  # 모델 실행
                loss = criterion(outputs, masks)  # ✅ CrossEntropyLoss 적용
                val_loss += loss.item()

                # IoU, Dice Score 계산
                iou = compute_iou(outputs, masks)
                dice = compute_dice(outputs, masks)
                iou_scores.append(iou)
                dice_scores.append(dice)


        avg_val_loss = val_loss / len(val_data)
        avg_iou = np.mean(iou_scores)
        avg_dice = np.mean(dice_scores)
        elapsed_time = time.perf_counter() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB 단위

        # 로그 기록
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_iou, avg_dice, elapsed_time, current_lr, gpu_memory])

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Time: {elapsed_time:.2f}s | LR: {current_lr:.6f} | GPU: {gpu_memory:.2f}MB")

    save_model(model, epochs, result_dir)
    print("Model saved!")
    print("Training complete!")

def model_train_multi(train_data, val_data, epochs, learning_rate, log_file, result_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=3, output_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "IoU", "Dice Score", "Time (s)", "Learning Rate", "GPU Memory (MB)"])

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.perf_counter()

        for images, masks in train_data:
            images = images.clone().detach().to(device, dtype=torch.float32)
            masks = masks.clone().detach().to(device, dtype=torch.long).squeeze(1)  # ✅ 변환

            optimizer.zero_grad()  # ✅ 그래디언트 초기화 추가
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_data)

        model.eval()
        val_loss = 0.0
        iou_scores = []
        dice_scores = []

        with torch.no_grad():
            for images, masks in val_data:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long).squeeze(1)  # ✅ 정수형 변환

                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                iou = compute_iou(outputs, masks)
                dice = compute_dice(outputs, masks)
                iou_scores.append(iou)
                dice_scores.append(dice)

        avg_val_loss = val_loss / len(val_data)
        avg_iou = np.mean(iou_scores)
        avg_dice = np.mean(dice_scores)
        elapsed_time = time.perf_counter() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        gpu_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB 단위

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_iou, avg_dice, elapsed_time, current_lr, gpu_memory])

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f} | Time: {elapsed_time:.2f}s | LR: {current_lr:.6f} | GPU: {gpu_memory:.2f}MB")

    save_model(model, epochs, result_dir)
    print("Model saved!")
    print("Training complete!")

# ✅ IoU & Dice Score 계산 함수


# ✅ IoU & Dice Score 계산 함수
def compute_iou_test(preds, targets):
    intersection = np.logical_and(preds, targets).sum()
    union = np.logical_or(preds, targets).sum()
    return intersection / union if union > 0 else 1.0

def compute_dice_test(preds, targets):
    intersection = np.logical_and(preds, targets).sum()
    return (2. * intersection) / (preds.sum() + targets.sum()) if (preds.sum() + targets.sum()) > 0 else 1.0

# ✅ 이미지 저장 함수 (IoU 시각화 + 범례 추가)
def save_result_image(idx, og, unet, target, p, iou_score, dice_score, save_path):
    fig, axs = plt.subplots(1, 4, figsize=(20, 12))

    # ✅ Ground Truth (정답 마스크) → 0번 위치
    axs[0].imshow(target, cmap="gray")
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    # ✅ IoU 시각화 → 1번 위치
    if target.ndim == 3 and target.shape[-1] == 1:
        target = target.squeeze(-1)  # (H, W, 1) → (H, W)

    intersection = np.logical_and(unet, target)
    union = np.logical_or(unet, target)

    iou_visual = np.zeros((*unet.shape, 3), dtype=np.uint8)  # (H, W, 3)
    iou_visual[target.astype(bool)] = [0, 255, 0]  # GT → 초록색
    iou_visual[unet.astype(bool)] = [0, 0, 255]  # 예측 → 파란색
    iou_visual[intersection.astype(bool)] = [255, 255, 255]  # 겹치는 부분 → 흰색

    axs[1].imshow(iou_visual)
    axs[1].set_title(f"IoU Visualization {idx}\nIoU: {iou_score:.4f} | Dice: {dice_score:.4f}")
    axs[1].axis("off")

    # ✅ 원본 이미지 → 2번 위치
    axs[2].imshow(og)
    axs[2].set_title(f"Original {idx}")
    axs[2].axis("off")

    # ✅ 예측 마스크 → 3번 위치
    axs[3].imshow(unet, cmap="gray")
    axs[3].set_title(f"U-Net Prediction (p > {p})")
    axs[3].axis("off")

    # ✅ 범례 추가 (Green = GT, Blue = Prediction, White = Intersection)
    green_patch = mpatches.Patch(color='green', label='Ground Truth (GT)')
    blue_patch = mpatches.Patch(color='blue', label='Predicted Mask')
    white_patch = mpatches.Patch(color='white', label='Intersection (IoU)')

    # ✅ 범례를 figure 전체에 추가 (좌하단 fig 밖에 위치)
    legend_fig = fig.legend(
        handles=[green_patch, blue_patch, white_patch],
        loc='lower left',  # 왼쪽 아래 (fig 바깥쪽)
        bbox_to_anchor=(-0.05, -0.05),  # fig 바깥쪽 위치 조정 (x, y)
        fontsize=12,
        framealpha=0.7,  # 투명도 설정
        edgecolor="black"  # 테두리 색상 추가
    )

    # ✅ 이미지 저장 (IoU & Dice Score 포함)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

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
def save_result_image3(idx, og, unet, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(20, 12))


    # ✅ 원본 이미지 → 2번 위치
    axs[0].imshow(og)
    axs[0].set_title(f"Original {idx}")
    axs[0].axis("off")

    # ✅ 예측 마스크 → 3번 위치
    axs[1].imshow(unet, cmap="gray")
    axs[1].set_title(f"U-Net Prediction")
    axs[1].axis("off")

    # ✅ 이미지 저장 (IoU & Dice Score 포함)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
# ✅ 테스트 데이터 로드 함수
def test_data_load(images_test):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images_test_torch = torch.tensor(images_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    images_test_torch = images_test_torch.to(device)
    return images_test_torch

# ✅ 모델 테스트 및 결과 저장 함수
# ✅ 모델 테스트 및 결과 저장 함수
def model_test(test_data, masks_test, images_test, epochs, Threshold, result_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=3, output_channels=1).to(device)
    load_model(model, epochs, result_dir)

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
                iou = compute_iou_test(unet_predictions[i][idx], masks_test[idx])
                dice = compute_dice_test(unet_predictions[i][idx], masks_test[idx])

                writer.writerow([idx, r, round(iou, 4), round(dice, 4)])

                save_path = os.path.join(result_dir, f"result_{idx}_threshold_{r}.png")
                save_result_image(idx, images_test[idx], unet_predictions[i][idx], masks_test[idx], r, iou, dice, save_path)
                print(f"[Index {idx}, Threshold {r}] IoU: {iou:.4f}, Dice Score: {dice:.4f}")

    np.save(os.path.join(result_dir, "unet_predictions.npy"), unet_predict)
    del model
    torch.cuda.empty_cache()
    print(f"✅ 테스트 결과 저장 완료! 폴더: {result_dir}")
#1레이어 1장
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

                save_path = os.path.join(result_dir, f"result_{idx}_threshold_{r}.png")
                save_result_image2(idx, images_test[idx], unet_predictions[i][idx], r, save_path)
                print(f"[Index {idx}, Threshold {r}]")

    np.save(os.path.join(result_dir, "unet_predictions.npy"), unet_predict)
    del model
    torch.cuda.empty_cache()
    print(f"✅ 테스트 결과 저장 완료! 폴더: {result_dir}")

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

def with_log(epochs, learning_rate, batch_size, apply_aug: bool, augmentation_factor,
             images_dir, masks_dir, image_size, test_size, val_size, timestamp, result_dir):

    log_file = f"{result_dir}/train_log_{epochs}epochs_{timestamp}.csv"
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['time:', timestamp, 'epochs:', epochs, 'learning_rate:', learning_rate,
                         'batch_size:', batch_size, 'Augmentation:', apply_aug, 'augmentation_factor:', augmentation_factor,
                         'images_dir:', images_dir, 'masks_dir:', masks_dir, 'image_size:', image_size,
                         'Train:Test:Val :', 1 - test_size, ':', test_size, ':', val_size])

        if apply_aug:
            writer.writerow(['Augmentation Applied', 'Transformations:', get_augmentation().__repr__()])

    # ✅ 데이터 로드
    images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data_multi(
        images_dir, masks_dir, image_size, test_size, val_size
    )

    # ✅ DataLoader 생성 (Augmentation 반영)
    train_loader = train_data_load(images_train, masks_train, batch_size, apply_aug, augmentation_factor)
    val_loader = val_data_load(images_val, masks_val, batch_size, apply_aug=False)  # 🚀 Validation에도 적용 가능

    # ✅ 모델 학습 실행
    model_train_3(train_loader, val_loader, epochs, learning_rate, log_file,result_dir)

def model_test3(test_data, images_test, epochs, model_dir, result_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(input_channels=3, output_channels=3).to(device)
    load_model(model, epochs, model_dir)

    # 모델을 평가 모드로 전환
    model.eval()

    # 모델 예측
    with torch.no_grad():
        unet_predict = model(test_data)
        unet_predict = torch.softmax(unet_predict, dim=1)  # 다중 클래스 예측을 위한 softmax 적용

    # NumPy 변환
    print('unet_predict.shape', unet_predict.shape)
    unet_predict = torch.argmax(unet_predict, dim=1).cpu().numpy()  # 가장 높은 확률의 클래스를 선택

    # CSV 저장 파일 생성
    csv_file = os.path.join(result_dir, "test_results.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Index", "IoU", "Dice Score"])  # Threshold 제거

        for idx in range(len(unet_predict)):
            writer.writerow([idx])  # Threshold 없이 저장

            save_path = os.path.join(result_dir, f"result_{idx}.png")
            save_result_image3(idx, images_test[idx], unet_predict[idx], save_path)
            print(f"[Index {idx}]")

    np.save(os.path.join(result_dir, "unet_predictions.npy"), unet_predict)
    del model
    torch.cuda.empty_cache()
    print(f"✅ 테스트 결과 저장 완료! 폴더: {result_dir}")


# # 🔹 학습된 모델 불러오기
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# images_dir = "./testdata/images"
# masks_dir = "./testdata/masks"
# batch_size = 16
# epochs = 200
# learning_rate = 0.0001
# image_size = 512
# test_size =0.2
# val_size = 0.2
# Augmentation = True
# augmentation_factor = 10
#
#
# # ✅ 결과 저장을 위한 폴더 생성
# result_dir = f"./unet/test_results{timestamp}"
# os.makedirs(result_dir, exist_ok=True)
#
# # 모델 학습
# with_log(epochs, learning_rate,batch_size,Augmentation,augmentation_factor,images_dir,masks_dir,image_size, test_size, val_size,timestamp,result_dir)
#
#
# # ✅ 데이터 로드
# images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data(images_dir, masks_dir, image_size, test_size, val_size)

# # ✅ Augmentation 켜고 실행
# visualize_augmentation(images_train, masks_train, apply_aug=True, num_samples=3)
#
#
# # ✅ Augmentation 끄고 실행
# visualize_augmentation(images_train, masks_train, apply_aug=False, num_samples=3)


# # ✅ 테스트 실행 (결과 저장 디렉토리 반영)
# Threshold = [0.4, 0.5, 0.6, 0.7]
# model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
# Threshold = [0.45, 0.55, 0.65, 0.75]
# model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)