from hair_Segmentation import *  # U-Net 모델 불러오기
# 🔹 학습된 모델 불러오기
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
images_dir = "./testdata/images"
masks_dir = "./testdata/masks"
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
images_train, images_val, images_test, masks_train, masks_val, masks_test = load_data_multi(images_dir, masks_dir, image_size, test_size, val_size)


# ✅ 테스트 실행 (결과 저장 디렉토리 반영)
Threshold = [0.4, 0.5, 0.6, 0.7]
model_test(test_data_load(images_test), masks_test, images_test, epochs, Threshold, result_dir)