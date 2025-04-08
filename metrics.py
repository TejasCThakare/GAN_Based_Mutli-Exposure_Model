import torch
import torchmetrics.functional as TF

def psnr(img1, img2):
    return TF.peak_signal_noise_ratio(img1, img2)

def ssim(img1, img2):
    return TF.structural_similarity_index_measure(img1, img2)
