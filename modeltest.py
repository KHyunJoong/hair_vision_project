from hair_Segmentation import *  # U-Net 모델 불러오기
import torch
images_dir = "./testdata/images"
masks_dir = "./testdata/masks"
# 🔹 학습된 모델 불러오기
#
# timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# batch_size = 4
# epochs = 10
# learning_rate = 0.001
# image_size = 512
# test_size =0.2
# val_size = 0.2
# Augmentation = False
# augmentation_factor = 0
#
#
# # ✅ 결과 저장을 위한 폴더 생성
# result_dir = f"./unet/test_results{timestamp}"
# os.makedirs(result_dir, exist_ok=True)
#
# # 모델 학습
# with_log(epochs, learning_rate,batch_size,Augmentation,augmentation_factor,images_dir,masks_dir,image_size, test_size, val_size,timestamp,result_dir)
#
# # ✅ 데이터 로드
# images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data(images_dir, masks_dir, image_size, test_size, val_size)
#
# # ✅ 테스트 실행 (결과 저장 디렉토리 반영)
# Threshold = [0.6, 0.65, 0.7, 0.75]
# model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
########################################################################################################################
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_size = 16
epochs = 10
learning_rate = 0.001
image_size = 512
test_size =0.2
val_size = 0.2
Augmentation = False
augmentation_factor = 0


# ✅ 결과 저장을 위한 폴더 생성
result_dir = f"./unet/test_results{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 모델 학습
with_log(epochs, learning_rate,batch_size,Augmentation,augmentation_factor,images_dir,masks_dir,image_size, test_size, val_size,timestamp,result_dir)

# ✅ 데이터 로드
images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data(images_dir, masks_dir, image_size, test_size, val_size)

# ✅ 테스트 실행 (결과 저장 디렉토리 반영)
Threshold = [0.6, 0.65, 0.7, 0.75]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
Threshold = [0.8, 0.85, 0.9, 0.99]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)


###############################################################################################

###############################################################################################
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_size = 16
epochs = 10
learning_rate = 0.001
image_size = 512
test_size =0.2
val_size = 0.2
Augmentation = True
augmentation_factor = 10


# ✅ 결과 저장을 위한 폴더 생성
result_dir = f"./unet/test_results{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 모델 학습
with_log(epochs, learning_rate,batch_size,Augmentation,augmentation_factor,images_dir,masks_dir,image_size, test_size, val_size,timestamp,result_dir)

# ✅ 데이터 로드
images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data(images_dir, masks_dir, image_size, test_size, val_size)

# ✅ 테스트 실행 (결과 저장 디렉토리 반영)
Threshold = [0.6, 0.65, 0.7, 0.75]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
Threshold = [0.8, 0.85, 0.9, 0.99]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
###############################################################################################

###############################################################################################
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_size = 4
epochs = 100
learning_rate = 0.001
image_size = 512
test_size =0.2
val_size = 0.2
Augmentation = False
augmentation_factor = 0


# ✅ 결과 저장을 위한 폴더 생성
result_dir = f"./unet/test_results{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 모델 학습
with_log(epochs, learning_rate,batch_size,Augmentation,augmentation_factor,images_dir,masks_dir,image_size, test_size, val_size,timestamp,result_dir)

# ✅ 데이터 로드
images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data(images_dir, masks_dir, image_size, test_size, val_size)

# ✅ 테스트 실행 (결과 저장 디렉토리 반영)
Threshold = [0.6, 0.65, 0.7, 0.75]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
Threshold = [0.8, 0.85, 0.9, 0.99]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
###############################################################################################

###############################################################################################
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_size = 16
epochs = 100
learning_rate = 0.001
image_size = 512
test_size =0.2
val_size = 0.2
Augmentation = False
augmentation_factor = 0


# ✅ 결과 저장을 위한 폴더 생성
result_dir = f"./unet/test_results{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 모델 학습
with_log(epochs, learning_rate,batch_size,Augmentation,augmentation_factor,images_dir,masks_dir,image_size, test_size, val_size,timestamp,result_dir)

# ✅ 데이터 로드
images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data(images_dir, masks_dir, image_size, test_size, val_size)

# ✅ 테스트 실행 (결과 저장 디렉토리 반영)
Threshold = [0.6, 0.65, 0.7, 0.75]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
Threshold = [0.8, 0.85, 0.9, 0.99]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
###############################################################################################

###############################################################################################
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
batch_size = 16
epochs = 100
learning_rate = 0.001
image_size = 512
test_size =0.2
val_size = 0.2
Augmentation = True
augmentation_factor = 10


# ✅ 결과 저장을 위한 폴더 생성
result_dir = f"./unet/test_results{timestamp}"
os.makedirs(result_dir, exist_ok=True)

# 모델 학습
with_log(epochs, learning_rate,batch_size,Augmentation,augmentation_factor,images_dir,masks_dir,image_size, test_size, val_size,timestamp,result_dir)

# ✅ 데이터 로드
images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data(images_dir, masks_dir, image_size, test_size, val_size)

# ✅ 테스트 실행 (결과 저장 디렉토리 반영)
Threshold = [0.6, 0.65, 0.7, 0.75]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)
Threshold = [0.8, 0.85, 0.9, 0.99]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)