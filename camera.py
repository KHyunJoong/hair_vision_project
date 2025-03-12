# import evdev
# import cv2
# import time
#
# # 이벤트 디바이스 찾기
# devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
# for device in devices:
#     print(f"Device found: {device.path} - {device.name}")
#
# # 카메라 설정
# cap = cv2.VideoCapture(0)  # USB 카메라
#
# # 버튼을 감지할 장치 선택 (보통 USB 카메라 버튼은 HID 장치)
# device_path = "/dev/input/eventX"  # 'X'를 실제 장치 번호로 변경
# device = evdev.InputDevice(device_path)
#
# print("Listening for button press...")
#
# # 이벤트 감지 루프
# for event in device.read_loop():
#     if event.type == evdev.ecodes.EV_KEY and event.value == 1:  # 버튼 눌림 감지
#         print("Button pressed! Capturing image...")
#         ret, frame = cap.read()
#         if ret:
#             filename = f"capture_{int(time.time())}.jpg"
#             cv2.imwrite(filename, frame)
#             print(f"Image saved: {filename}")

pip install pyusb opencv-python
