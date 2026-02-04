import torch
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn as nn
import cv2
import os
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure
class Pix2PixModel(BaseModel):
    def name(self):
        return 'Pix2PixModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(pool_size=0, no_lsgan=True, norm='batch')
        parser.set_defaults(dataset_mode='aligned')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.loss_names = ['G_GAN', 'G_L1', 'G_GD', 'D_real', 'D_fake']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG = self.netG.to(self.device)


        self.sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32, device=self.device)
        self.sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32, device=self.device)

        if self.isTrain:

            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm,True, opt.init_type, opt.init_gain, self.gpu_ids)
    
        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
   
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']



    def edge_aware_smoothness_loss_with_sobel(self, img):
        img = img.to(dtype=torch.float32)
        edge_x = torch.nn.functional.conv2d(img, self.sobel_x, padding=1)
        edge_y = torch.nn.functional.conv2d(img, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        

        if torch.isnan(edge_magnitude).any() or torch.isinf(edge_magnitude).any():
            print("edge_magnitude contains NaN or Inf values!")
        
        dx = torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])
        dy = torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
        edge_weight_x = torch.exp(-torch.clamp(edge_magnitude[:, :, 1:, :], min=0, max=10))  
        edge_weight_y = torch.exp(-torch.clamp(edge_magnitude[:, :, :, 1:], min=0, max=10))  
        

        if torch.isnan(edge_weight_x).any() or torch.isinf(edge_weight_x).any():
            print("edge_weight_x contains NaN or Inf values!")
        if torch.isnan(edge_weight_y).any() or torch.isinf(edge_weight_y).any():
            print("edge_weight_y contains NaN or Inf values!")
        
        return (dx * edge_weight_x).mean() + (dy * edge_weight_y).mean()

    def gradient_difference_loss(self,fake_B, real_B):
        grad_real_x = torch.abs(real_B[:, :, 1:, :] - real_B[:, :, :-1, :])
        grad_real_y = torch.abs(real_B[:, :, :, 1:] - real_B[:, :, :, :-1])
        grad_fake_x = torch.abs(fake_B[:, :, 1:, :] - fake_B[:, :, :-1, :])
        grad_fake_y = torch.abs(fake_B[:, :, :, 1:] - fake_B[:, :, :, :-1])
        loss = torch.mean(torch.abs(grad_real_x - grad_fake_x)) + torch.mean(torch.abs(grad_real_y - grad_fake_y))
        return loss


    def forward(self):
        self.fake_B = self.netG(self.real_A).to(self.device)

    def backward_D(self):

        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)  

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)  

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = 300 * self.criterionL1(self.fake_B, self.real_B)

 
        if self.epoch > 200:
            self.loss_G_GD = 100 * self.gradient_difference_loss(self.fake_B, self.real_B)
        else:
            self.loss_G_GD = 0


        output_dir = 'output_images'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

     
        self.loss_G = self.loss_G_GAN + self.loss_G_L1  + self.loss_G_GD 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def set_epoch(self, epoch):
        self.epoch = epoch
