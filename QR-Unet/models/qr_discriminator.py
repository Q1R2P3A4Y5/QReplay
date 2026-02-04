import torch
import cv2
from cv2.wechat_qrcode import WeChatQRCode
from .networks import NLayerDiscriminator
import os
import torch.nn as nn

class QRDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, n_layers, norm_layer, use_sigmoid):
        super(QRDiscriminator, self).__init__()

        self.base_discriminator = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid)

        model_dir = "models/WeChat_QR_Model"
        self.qr_decoder = WeChatQRCode(
            detector_prototxt_path=os.path.join(model_dir, "detect.prototxt"),
            detector_caffe_model_path=os.path.join(model_dir, "detect.caffemodel"),
            super_resolution_prototxt_path=os.path.join(model_dir, "sr.prototxt"),
            super_resolution_caffe_model_path=os.path.join(model_dir, "sr.caffemodel")
        )

    def forward(self, x):
       
        validity = self.base_discriminator(x)


        decoded_results = []
        for img in x.detach().cpu().numpy(): 
            img = img.squeeze() * 255 
            img = img.astype('float32')  
            
 
            img = cv2.convertScaleAbs(img)  

            decoded_text, points = self.qr_decoder.detectAndDecode(img)
            decoded_results.append(1.0 if decoded_text else 0.0)

        decoded_results = torch.tensor(decoded_results, device=x.device).unsqueeze(-1)
        return validity, decoded_results
