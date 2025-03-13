from ultralytics import YOLO

model = YOLO("hair/hair_model4/weights/best.pt")

# 모델을 'yolo200_float32.tflite'로 변환하여 저장
model.export(format='tflite', name="yolo200_float32")
