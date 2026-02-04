import os
import torch
import cv2
import numpy as np
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from cv2.wechat_qrcode import WeChatQRCode
import time

if __name__ == '__main__':
    opt = TestOptions().parse()
    # 固定测试相关的参数
    opt.num_threads = 1   # 测试代码仅支持单线程
    opt.batch_size = 1    # 测试代码仅支持 batch_size=1
    opt.serial_batches = True  # 禁止数据乱序加载
    opt.no_flip = True    # 禁止图像翻转
    opt.display_id = -1   # 不使用 visdom 显示结果
    opt.dataset_model = 'aligned'
    opt.input_nc = 1
    opt.output_nc = 1
    opt.resize_or_crop = 'scale_width'
    opt.model = 'pix2pix'
    opt.netG = 'unet_256'
    # opt.netG = 'resnet_9blocks'
    # opt.dataroot = 'datasets/qrcodes/AB'

    print(f"Testing data from: {opt.dataroot}")

    # 创建数据加载器
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()

    # 创建模型
    model = create_model(opt)
    model.setup(opt)

    # 创建用于保存结果的 HTML 页面
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    # 初始化二维码解码器
    model_dir = "models/WeChat_QR_Model"
    qr_decoder = WeChatQRCode(
        detector_prototxt_path=os.path.join(model_dir, "detect.prototxt"),
        detector_caffe_model_path=os.path.join(model_dir, "detect.caffemodel"),
        super_resolution_prototxt_path=os.path.join(model_dir, "sr.prototxt"),
        super_resolution_caffe_model_path=os.path.join(model_dir, "sr.caffemodel")
    )

    # 统计二维码检测成功的次数
    total_images = 0
    detected_qr_codes_original = 0  # 原始图像的检测成功计数
    detected_count = 0   # 缩放图像的检测成功计数
    detected_qr_codes_test = 0
    # 如果指定了 eval 模式，则启用模型的评估模式
    if opt.eval:
        model.eval()

    total_processing_time = 0

    # 遍历数据集并进行测试
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        start_time = time.time()
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()  # 获取模型输出的图像
        img_path = model.get_image_paths()    # 获取图像路径

        # 保存结果图像
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        end_time = time.time()
        total_processing_time += end_time - start_time

    # 保存结果页面
    webpage.save()
    print(f"Results saved in: {web_dir}")

    # 左上角
    locator1 = np.array([
        [0, 0, 0, 0, 0, 0, 0, 255],
        [0, 255, 255, 255, 255, 255, 0, 255],
        [0, 255, 0, 0, 0, 255, 0, 255],
        [0, 255, 0, 0, 0, 255, 0, 255],
        [0, 255, 0, 0, 0, 255, 0, 255],
        [0, 255, 255, 255, 255, 255, 0, 255],
        [0, 0, 0, 0, 0, 0, 0, 255],
        [255, 255, 255, 255, 255, 255, 255, 255]
    ])

    # 右上角
    locator2 = np.array([
        [255, 0, 0, 0, 0, 0, 0, 0],
        [255, 0, 255, 255, 255, 255, 255, 0],
        [255, 0, 255, 0, 0, 0, 255, 0],
        [255, 0, 255, 0, 0, 0, 255, 0],
        [255, 0, 255, 0, 0, 0, 255, 0],
        [255, 0, 255, 255, 255, 255, 255, 0],
        [255, 0, 0, 0, 0, 0, 0, 0],
        [255, 255, 255, 255, 255, 255, 255, 255]
    ])

    # 左下角
    locator3 = np.array([
        [255, 255, 255, 255, 255, 255, 255, 255],
        [0, 0, 0, 0, 0, 0, 0, 255],
        [0, 255, 255, 255, 255, 255, 0, 255],
        [0, 255, 0, 0, 0, 255, 0, 255],
        [0, 255, 0, 0, 0, 255, 0, 255],
        [0, 255, 0, 0, 0, 255, 0, 255],
        [0, 255, 255, 255, 255, 255, 0, 255],
        [0, 0, 0, 0, 0, 0, 0, 255]
    ])

    locator4 = np.array([[0, 255, 0, 255, 0, 255, 0, 255, 0]])

    locator5 = np.array([[0], [255], [0], [255], [0], [255], [0], [255], [0]]).flatten()

    locator_small = np.array([
        [0, 0, 0, 0, 0],
        [0, 255, 255, 255, 0],
        [0, 255, 0, 255, 0],
        [0, 255, 255, 255, 0],
        [0, 0, 0, 0, 0]
    ])

    # 开始遍历生成的 fake 图像文件
    fake_images_dir = os.path.join(web_dir, 'images')
    fake_images = [f for f in os.listdir(fake_images_dir) if 'fake_B' in f and f.endswith('.png')]

    for fake_image in fake_images:
        image_path = os.path.join(fake_images_dir, fake_image)
        start_time = time.time()

        # 加载原始图像进行解码
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像有效
        if img is None:
            print(f"Failed to load image: {image_path}")
            continue

        # 原始图像解码
        decoded_text, points = qr_decoder.detectAndDecode(img)
        if decoded_text:
            detected_qr_codes_original += 1  # 原始图像解码成功
            print(f"Original: {fake_image}")

        total_images += 1

        # 对图像进行缩放操作
        img_resized = cv2.resize(img, (25, 25), interpolation=cv2.INTER_LANCZOS4)

        # 将定位符矩阵插入到相应位置
        # 左上角
        img_resized[:8, :8] = locator1
        # 右上角
        img_resized[:8, -8:] = locator2
        # 左下角
        img_resized[-8:, :8] = locator3

        # (9,7) (16,7) 水平线
        img_resized[6, 8:17] = locator4
        # (7,9) (7,18) 垂直线
        img_resized[8:17, 6] = locator5

        pooling_image_path = image_path.replace('_B', '_Pooling')
        img_resized = cv2.resize(img_resized, (250, 250), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(pooling_image_path, img_resized)

        # 缩放图像解码
        decoded_text_resized, points_resized = qr_decoder.detectAndDecode(img_resized)
        if decoded_text_resized:
            detected_count += 1  # 缩放图像解码成功
            print(f"Detected: {fake_image}")
        else:
            # 原始图像解码
            decoded_text, points = qr_decoder.detectAndDecode(img)
            if decoded_text:
                detected_count += 1  # 缩放图像解码成功
                print(f"Original: {fake_image}")
            else:
                print(f'Not Detected!  {fake_image}')

        end_time = time.time()
        total_processing_time += end_time - start_time

    # 计算原始图像的检测率
    # detection_rate_original = detected_qr_codes_original / total_images if total_images > 0 else 0
    # print(f"Original Detection Rate: {detection_rate_original * 100:.2f}%")

    # 计算缩放图像的检测率
    detection_rate_resized = detected_count / total_images if total_images > 0 else 0
    print(
        f"Detection Rate: {detection_rate_resized * 100:.2f}%\n"
        f"Total Images:   {total_images}\n"
        f"Detected:       {detected_count}"
    )

    if total_images > 0:
        average_processing_time = total_processing_time / total_images
        print(f"Average processing time per image: {average_processing_time:.4f} seconds")
    else:
        print("No images were processed.")
