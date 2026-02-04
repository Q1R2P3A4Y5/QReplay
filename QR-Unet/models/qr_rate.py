import cv2
import os
from cv2.wechat_qrcode import WeChatQRCode
from tqdm import tqdm

model_dir = "WeChat_QR_Model"
qr_decoder = WeChatQRCode(
    detector_prototxt_path=os.path.join(model_dir, "detect.prototxt"),
    detector_caffe_model_path=os.path.join(model_dir, "detect.caffemodel"),
    super_resolution_prototxt_path=os.path.join(model_dir, "sr.prototxt"),
    super_resolution_caffe_model_path=os.path.join(model_dir, "sr.caffemodel")
)

detected_qr_codes = 0
total_images = 0

fake_images_dir = os.path.join('/home/luzhenwei/chl/0pix2pix/pix2pix_qrcode_modify/results/qr1_1208/test_latest', 'images')
fake_images = [f for f in os.listdir(fake_images_dir) if 'fake_B' in f and f.endswith('.png')]


for fake_image in tqdm(fake_images):

    image_path = os.path.join(fake_images_dir, fake_image)

    img = cv2.imread(image_path)

    if img is None:
        print(f"Failed to load image: {image_path}")
        continue

    decoded_text, points = qr_decoder.detectAndDecode(img)

    if decoded_text:
        detected_qr_codes += 1  
        print(fake_image)
    total_images += 1

detection_rate = detected_qr_codes / total_images if total_images > 0 else 0
print(f"Detection  for 'fake' images: {detected_qr_codes} / {total_images}")
print(f"Detection Rate for 'fake' images: {detection_rate * 100:.2f}%")