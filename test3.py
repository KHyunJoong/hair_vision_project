import torch

# print("CUDA 사용 가능 여부:", torch.cuda.is_available())
# print("사용 가능한 GPU 개수:", torch.cuda.device_count())
# print("사용 중인 GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU 없음")

from ultralytics import YOLO
import torch
# 베이스 모델 로드
if __name__ == '__main__':
    model = YOLO("yolo11n.pt").to("cuda")

    # 모델 학습
    model.train(data='datasets/hair/hair.yaml',
                epochs=10,
                project='hair',
                name='hair_model',
                device='cuda')
