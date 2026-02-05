# QReplay

QReplay is an integrated QR code processing framework that combines a Matlab-based **DCD reconstruction module** and a PyTorch-based **QR-Unet enhancement module**.

---

## Prerequisites

- Linux or macOS
- Python 3.X
- Matlab 2019a
- CPU or NVIDIA GPU with CUDA and cuDNN

---

## Getting Started

## Step 1 — DCD 

The DCD module is used for QR code digital content decomposition and initial reconstruction.

**Requirement:** Matlab 2019a

**Run the demo**
```matlab
main_recon_demo
```

##  Step 2 — QR-Unet 
### Installation

Install PyTorch 0.4+ and torchvision from http://pytorch.org and other dependencies (e.g., visdom and dominate). You can install all the dependencies by
```
pip install -r requirements.txt
```
### Train a Model
Start training with:
```
python train.py \
  --dataroot ./qrcodes-mix/datasets/train \
  --name qrcodes_blur2 \
  --niter 200 \
  --niter_decay 100 \
  --batch_size 256
```
### Test the Model

Run batch testing with test2.py. Replace the experiment name with your trained model name:
```
python test2.py \
  --dataroot ./qrcodes-mix/datasets/test \
  --model pix2pix \
  --name your weight name
```
## Disclaimer

### Use Restriction
This project, including all associated code and models, is provided **solely for academic research and educational purposes**. Any unlawful use, misuse, or application that may violate applicable laws, regulations, or the privacy rights of individuals or organizations is **strictly prohibited**.

### No Warranty
This project is provided **“as is”** without warranties of any kind, whether express or implied, including but not limited to warranties of accuracy, completeness, reliability, fitness for a particular purpose, or non-infringement. Users assume full responsibility for any direct or indirect consequences arising from the use of this project, including (but not limited to) data loss, system damage, service interruption, or legal disputes.

### Copyright Notice
All third-party open-source code, datasets, or resources referenced in this project remain the property of their respective owners. Users must comply with the corresponding licenses and attribution requirements.

### Data Ethics and Privacy Statement
For data generation and model training, we used a **synthetically generated** dataset of QR code images containing **18-digit numeric strings** that are designed to resemble typical payment-code formats. The dataset **does not include real identities, real payment credentials, or any private/sensitive information**. We conducted this research to evaluate methodological feasibility under realistic assumptions while maintaining strict compliance with ethical principles, data-privacy standards, and all applicable regulations. We emphasize legality, transparency, and social responsibility, and aim to protect the legitimate interests of all stakeholders.

### Final Interpretation Right
To the maximum extent permitted by law, the authors reserve the final right of interpretation of this disclaimer.
