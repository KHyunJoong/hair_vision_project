import threading
import time
import cv2
# import tflite_runtime.interpreter as tflite
import tensorflow as tf
import numpy as np
from collections import deque

# 첫 번째 모델 (hl_condition.py 대체)
class_names_condition = ['normal', 'mild', 'moderate', 'severe']

# 두 번째 모델 (hair_obd.py 대체)

class_names_hair = ['1hair', '2hair', '3hair', '4hair']

def hl_condition_script(img_q, result_q):
    model_path = './hlcM_torchQ16.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    while True:
        if not img_q:
            continue
        frame = img_q.popleft()
        image = cv2.resize(frame, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.array(image, dtype=np.float32)
        image = np.expand_dims(image, axis=0)

        # 추론
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        pred_ind = np.argmax(output_data[0])

        result_q.append(f'HL Condition Predicted class: {class_names_condition[pred_ind]}')

# 두 번째 모델 (hair_obd.py 대체)
def iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def merge_boxes(boxes, scores, classes, image_width, image_height, threshold=0.5):
    merged_result = []
    while boxes:
        score = scores.pop(0)
        cls = classes.pop(0)
        box = boxes.pop(0)
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * image_width)
        y1 = int((y_center - height / 2) * image_height)
        x2 = int((x_center + width / 2) * image_width)
        y2 = int((y_center + height / 2) * image_height)

        current_box = [x1, y1, x2, y2]
        to_merge = []
        for i, other_box in enumerate(boxes):
            x_center, y_center, width, height = other_box
            x1 = int((x_center - width / 2) * image_width)
            y1 = int((y_center - height / 2) * image_height)
            x2 = int((x_center + width / 2) * image_width)
            y2 = int((y_center + height / 2) * image_height)
            other_box_coords = [x1, y1, x2, y2]

            if iou(current_box, other_box_coords) > threshold:
                to_merge.append((i, other_box_coords))

        for i, merge_box in reversed(to_merge):
            boxes.pop(i)
            s = scores.pop(i)
            c = classes.pop(i)
            current_box[0] = min(current_box[0], merge_box[0])
            current_box[1] = min(current_box[1], merge_box[1])
            current_box[2] = max(current_box[2], merge_box[2])
            current_box[3] = max(current_box[3], merge_box[3])
            if s > score:
                score = s
                cls = c

        merged_result.append((current_box, score, cls))
    return merged_result

def hair_obd_script(img_q, result_q):
    model_path = 'yolo200_float32.tflite'
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    image_height = input_details[0]['shape'][1]
    image_width = input_details[0]['shape'][2]

    while True:
        if not img_q:
            continue
        image = img_q.popleft()
        image = cv2.resize(image, (image_width, image_height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_np = np.array(image)
        image_np = np.true_divide(image_np, 255, dtype=np.float32)
        image_np = image_np[np.newaxis, :]

        interpreter.set_tensor(input_details[0]['index'], image_np)

        start = time.time()
        interpreter.invoke()
        print(f'Run time: {time.time() - start:.2f}s')

        output = interpreter.get_tensor(output_details[0]['index'])
        output = output[0]
        output = output.T

        boxes_xywh = output[:, :4]
        scores = np.max(output[:, 4:], axis=1)
        classes = np.argmax(output[:, 4:], axis=1)

        threshold = 0.2
        filtered_boxes = []
        filtered_scores = []
        filtered_classes = []
        for i, s in enumerate(scores):
            if s > threshold:
                filtered_boxes.append(boxes_xywh[i])
                filtered_scores.append(s)
                filtered_classes.append(classes[i])

        # Merge boxes with updated parameters
        merged_result = merge_boxes(filtered_boxes, filtered_scores, filtered_classes, image_width, image_height)

        colormap = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        result = []
        for box, score, cls in merged_result:
            x1, y1, x2, y2 = box
            color = colormap[cls]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            text = f"{class_names_hair[cls]}"
            cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            result.append({'class': class_names_hair[cls], 'box': box})

        result_q.append((image, result))

# 메인 실행 함수
def run_scripts():
    img_q_condition = deque(maxlen=3)
    result_q_condition = deque(maxlen=10)

    img_q_hair = deque(maxlen=3)
    result_q_hair = deque(maxlen=10)

    # 두 스레드를 시작
    thread_a = threading.Thread(target=hl_condition_script, args=(img_q_condition, result_q_condition))
    thread_b = threading.Thread(target=hair_obd_script, args=(img_q_hair, result_q_hair))
    thread_c = threading.Thread(target=run_scripts)

    thread_a.start()
    thread_b.start()

    # 카메라 캡처
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    h, w, c = frame.shape
    cen = w // 2
    ofs = h // 2
    left, right = cen - ofs, cen + ofs

    while True:
        ret, frame = cap.read()
        img_q_condition.append(frame)
        img_q_hair.append(frame)

        if result_q_condition:
            condition_result = result_q_condition.popleft()
            print(condition_result)

        if result_q_hair:
            hair_result_image, hair_result = result_q_hair.popleft()
            print(f'Detected hair: {hair_result}')
            hair_result_image = cv2.cvtColor(hair_result_image, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hair OBD', hair_result_image)

        cv2.imshow('HL Condition', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    thread_a.join()
    thread_b.join()
    cv2.destroyAllWindows()

# 메인 함수 실행
if __name__ == "__main__":
    run_scripts()