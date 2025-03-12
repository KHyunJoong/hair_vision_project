import cv2
import numpy as np

def auto_canny(image, sigma=1):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    v = np.median(blurred)
    lower = int(max(30, (1.0 - sigma) * v))
    upper = int(min(50, (1.0 + sigma) * v))
    edged = cv2.Canny(blurred, lower, upper)
    return edged

# 이미지 로드
image = cv2.imread("testcut.jpg")
if image is None:
    print("이미지를 로드할 수 없습니다.")
else:
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 엣지 검출
    edges = auto_canny(gray)
    # 엣지 이미지를 컬러로 변환하여 원본과 겹치기 위해 3채널로 만듦
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # 원본 이미지와 엣지 이미지를 겹침
    combined = cv2.addWeighted(image, 0.8, edges_colored, 0.2, 0)

    # 이미지들을 수평으로 나열
    result = np.hstack((image, edges_colored, combined))

    # 결과 표시
    cv2.imshow("Original | Edges | Combined", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 이미지 불러오기 및 전처리
# image = cv2.imread("testcut.jpg", cv2.IMREAD_GRAYSCALE)
# # image_path = "일단 한 장만"  # 이미지 경로 설정
# # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 흑백 변환
#
# # 노이즈 제거 & 대비 증가
# image = cv2.GaussianBlur(image, (5, 5), 0)  # 가우시안 블러 적용
# image = cv2.equalizeHist(image)  # 히스토그램 평활화 (대비 향상)
#
# # Canny Edge Detection 적용 (엣지 검출)
# edges = cv2.Canny(image, threshold1=100, threshold2=100)
#
# # Hough Line Transform을 이용해 선 검출
# lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=10, maxLineGap=5)
#
# # 선의 평균 두께 계산
# thickness_list = []
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         thickness_list.append(abs(y2 - y1))  # Y축 기준으로 두께 측정
#
# # 평균 두께 출력
# if thickness_list:
#     avg_thickness = np.mean(thickness_list)
#     print(f"머리카락 평균 두께 (픽셀 단위): {avg_thickness:.2f} px")
# else:
#     print("머리카락을 감지하지 못했습니다.")
#
# # 결과 시각화 (원본 이미지 + 감지된 선)
# output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 1)
#
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1), plt.imshow(edges, cmap="gray"), plt.title("Canny Edge Detection")
# plt.subplot(1, 2, 2), plt.imshow(output_image), plt.title("Hough Line Transform")
# plt.show()
