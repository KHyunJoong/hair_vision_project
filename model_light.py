# import torch
#
# model_name = 'waste6M'
# model = torch.load(f'ptmodel/{model_name}.pt').to('cpu')
# dummy_input = torch.randn(1, 3, 224, 224) # 예: 이미지 분류 모델의 경우
# print(dummy_input)
# # 모델을 평가 모드로 설정
# model.eval()
#
# # PyTorch 모델을 ONNX 형식으로 변환 및 저장
# torch.onnx.export(model, dummy_input, "/waste_model.onnx", export_params=True,
#                   opset_version=18,
#                   input_names=['input'], output_names=['output'],
#                   dynamic_axes={'input' : {0 : 'batch_size'},
#                                 'output' : {0 : 'batch_size'}})
#
# onnx2tf -i /unet_model_200_dynamic_quantized.onnx

import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import csv

# ✅ TensorFlow Lite 모델 로드
TFLITE_MODEL_PATH = "unet_model_200_dynamic_quantized_float32.tflite"

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# ✅ 입력 및 출력 텐서 정보 가져오기
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ 입력 이미지 전처리 함수
def preprocess_image(image_path, image_size=512):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
    image = image.astype(np.float32) / 255.0  # 정규화
    image = np.expand_dims(image, axis=0)  # 배치 차원 추가
    return image

# ✅ 모델 실행 및 추론 함수
def run_inference(image_path, threshold=0.65):
    # 이미지 전처리
    image = preprocess_image(image_path)

    # 모델 입력 설정
    interpreter.set_tensor(input_details[0]['index'], image)

    # 추론 실행
    interpreter.invoke()

    # 결과 가져오기
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_mask = (output_data[0, :, :, 0] > threshold).astype(np.uint8)  # Threshold 적용

    # ✅ Threshold 적용 후 픽셀 개수 계산
    above_threshold = np.sum(output_mask > 0)  # 1로 변환된 픽셀 개수
    below_threshold = np.sum(output_mask == 0) # 0인 픽셀 개수

    return image[0], output_mask, above_threshold + below_threshold, above_threshold

# ✅ 결과 저장 함수
def save_result_image(idx, original, mask, save_path):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 원본 이미지
    axs[0].imshow(original)
    axs[0].set_title(f"Original {idx}")
    axs[0].axis("off")

    # 예측 마스크
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Predicted Mask")
    axs[1].axis("off")

    # 결과 저장
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# ✅ 테스트 이미지 폴더 및 결과 저장 폴더 설정
images_dir = "./datasets/hair/images/validation3"
result_dir = "./unet/test3_tflite"
os.makedirs(result_dir, exist_ok=True)

# ✅ CSV 파일 생성
csv_file = os.path.join(result_dir, "test_results.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Index", "Threshold", "Total Pixels", "Above Threshold Pixels"])

# ✅ 테스트 수행
test_images = sorted(os.listdir(images_dir))
threshold = 0.65  # 임계값 설정

for idx, image_name in enumerate(test_images):
    image_path = os.path.join(images_dir, image_name)

    try:
        original, mask, total_pixels, above_threshold = run_inference(image_path, threshold)
        save_path = os.path.join(result_dir, f"result_{idx}.png")
        save_result_image(idx, original, mask, save_path)

        # ✅ CSV에 결과 저장
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([idx, threshold, total_pixels, above_threshold])

        print(f"✅ 저장 완료: {save_path}, 총 픽셀: {total_pixels}, 임계값 초과 픽셀: {above_threshold}")

    except Exception as e:
        print(f"❌ 오류 발생: {image_path} - {e}")


print(f"✅ 모든 테스트 완료! 결과 저장 폴더: {result_dir}")

