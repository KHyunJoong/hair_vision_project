import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 이미지 불러오기
image = cv2.imread("./datasets/hair/images/validation2")
image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # LAB 색공간 변환

# 이미지 픽셀을 2D 배열로 변환
pixels = image.reshape((-1, 3))

# K-Means 클러스터링 (머리카락=0, 두피=1로 클러스터 2개 설정)
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(pixels)

# 가장 밝은 값(두피)을 1, 어두운 값(머리카락)을 0으로 정렬
label_0_mean = np.mean(pixels[labels == 0])  # 클러스터 0 평균 색상 값
label_1_mean = np.mean(pixels[labels == 1])  # 클러스터 1 평균 색상 값

if label_0_mean > label_1_mean:
    labels = np.where(labels == 0, 1, 0)  # 두피 = 1, 머리카락 = 0으로 변환
else:
    labels = np.where(labels == 1, 1, 0)  # 반대 상황 처리

# 원본 형태로 복원
segmented_image = labels.reshape(image.shape[:2])

# 머리카락과 두피 면적(픽셀 수) 계산
hair_pixels = np.sum(segmented_image == 0)  # 머리카락 픽셀 수
scalp_pixels = np.sum(segmented_image == 1)  # 두피 픽셀 수
total_pixels = hair_pixels + scalp_pixels  # 전체 픽셀 수

hair_ratio = hair_pixels / total_pixels * 100  # 머리카락 비율 (%)
scalp_ratio = scalp_pixels / total_pixels * 100  # 두피 비율 (%)

# 면적 정보 출력
print(f"머리카락 픽셀 수: {hair_pixels}개 ({hair_ratio:.2f}%)")
print(f"두피 픽셀 수: {scalp_pixels}개 ({scalp_ratio:.2f}%)")

# 결과 시각화 (색상 적용)
plt.figure(figsize=(8, 6))
plt.imshow(segmented_image, cmap='coolwarm')  # 컬러맵 적용 (두피=빨강, 머리카락=파랑)
plt.title("머리카락(0) vs 두피(1) 클러스터링 결과")
plt.colorbar(label="Cluster Value")
plt.show()