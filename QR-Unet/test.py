import os
import torch
import cv2
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import save_images
from util import html
from cv2.wechat_qrcode import WeChatQRCode

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
    opt.netG = 'resnet_9blocks'
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
    detected_qr_codes = 0

    # 如果指定了 eval 模式，则启用模型的评估模式
    if opt.eval:
        model.eval()

    # 遍历数据集并进行测试
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()  # 获取模型输出的图像
        img_path = model.get_image_paths()    # 获取图像路径

        # 保存结果图像
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)

    # 保存结果页面
    webpage.save()
    print(f"Results saved in: {web_dir}")


