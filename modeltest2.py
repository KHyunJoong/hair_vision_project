from hair_Segmentation import *  # U-Net 모델 불러오기
import torch
images_dir = "./datasets/hair/images/validation3"
model_dir = "./unet/test_results2025-03-10_02-30-26"
result_dir = "./unet/test2"
epochs = 100
Threshold = [0.40,0.45, 0.50,0.55, 0.60,0.65, 0.70,0.75,0.80,0.85,0.90,0.95]
image_size = 512
# ✅ 데이터 로드
images_test= load_data2(images_dir, image_size)
model_test3(test_data_load(images_test), images_test, epochs, model_dir, result_dir)

