import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import glob
import imageio
import natsort
from pytorch_msssim import ssim
import os
import os.path
import multiprocessing
import scipy.io as scio
from PIL import Image
import cv2
import matplotlib
import scipy.misc
from torchvision.utils import save_image

from model.myNet import *
from utils.metrics import Evaluator


def load_mri_to_do_test(dataset):
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    test_mri = np.zeros((len(data), image_width, image_length))
    for i in range(len(data)):
        #test_mri[i, :, :] = (imageio.imread(data[i]))
        test_mri[i, :, :] = np.array(Image.open(data[i]).convert('L'))
        test_mri[i, :, :] = (test_mri[i, :, :] - np.min(test_mri[i, :, :])) / (
                    np.max(test_mri[i, :, :]) - np.min(test_mri[i, :, :]))
        test_mri[i, :, :] = np.float32(test_mri[i, :, :])

    # expand dimension to add the channel
    test_mri = np.expand_dims(test_mri, axis=1)

    # convert the MRI Testing data to pytorch tensor
    test_mri_tensor = torch.from_numpy(test_mri).float()
    test_mri_tensor = test_mri_tensor.to(device)
    print(test_mri_tensor.shape)
    test_mri_tensor.requires_grad = True

    return test_mri_tensor


def load_pet_to_do_test(dataset):
    data = glob.glob(os.path.join(dataset, "*.*"))
    data = natsort.natsorted(data, reverse=False)
    pet_channels=4
    train_other = np.zeros((len(data), image_width, image_length, pet_channels), dtype=float)
    test_pet = np.zeros((len(data), image_width, image_length), dtype=float)
    for i in range(len(data)):
        # train_pet[i, :, :] = (imageio.imread(data[i]))
        test_pet[i, :, :] = np.array(Image.open(data[i]).convert('L'))
        test_pet[i, :, :] = (test_pet[i, :, :] - np.min(test_pet[i, :, :])) / (
                    np.max(test_pet[i, :, :]) - np.min(test_pet[i, :, :]))
        test_pet[i, :, :] = np.float32(test_pet[i, :, :])
    # expand dimension to add the channel
    test_pet = np.expand_dims(test_pet, axis=1)

    # convert the PET Testing data to pytorch tensor
    test_pet_tensor = torch.from_numpy(test_pet).float()
    test_pet_tensor = test_pet_tensor.to(device)
    print(test_pet_tensor.shape)
    test_pet_tensor.requires_grad = True

    return test_pet_tensor




if __name__=="__main__":
    # device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print(device)

    image_length = 256
    image_width = 256

    # load the model
    dtn = net()
    # dtn.load_state_dict(torch.load('model/MineCheckpoint.pth'))  # MineCheckpoint
    dtn.load_state_dict(torch.load('model/MineCheckpoint.pth', map_location='cpu'))
    dtn.eval()

    # load image
    DataSet = 'MRI-CT'   #change the type of fusion ('MRI-SPECT' 'MRI-PET' 'MRI-CT' )
    # dataset1 = 'testImages/MRI_SPECT/MRI/'
    # dataset1 = 'testImages/MRI_PET/MRI/'
    dataset1 = 'testImages/MRI_CT/MRI/'
    # dataset2 = 'testImages/MRI_SPECT/SPECT/'
    # dataset2 = 'testImages/MRI_PET/PET/'
    dataset2 = 'testImages/MRI_CT/CT/'
    test_mri_tensor = load_mri_to_do_test(dataset1)
    test_pet_tensor = load_pet_to_do_test(dataset2)

    # predicted the fused image
    fused = dtn(test_mri_tensor.to(device), test_pet_tensor.to(device))
    fused_numpy = fused.data.cpu().numpy()
    savePath='testResults'
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    if DataSet=='MRI-SPECT':
        metric_result = np.zeros((4))
        save_metrics = open(savePath+'/4metrics_74MR-SPECT.txt', 'a')
        save_metrics.write('i' + ' ' + 'VIFF' + ' ' + 'SSIM' + ' ' + 'CC' + ' ' + 'MSE' + '\n')
        for i in range(test_mri_tensor.shape[0]):
            fi = fused_numpy[i, 0, :, :]
            name = str(i)
            newPath = savePath+'/MR-SPECT_74/'
            p0 = newPath + 'originalIms/' + name
            p1 = newPath + 'fusedIms/' + name
            if not os.path.exists(p0):
                os.makedirs(p0)
            if not os.path.exists(p1):
                os.makedirs(p1)

            save_image(test_mri_tensor[i, 0, :, :], p0 + '/mri_' + name + '.png')
            save_image(test_pet_tensor[i, 0, :, :], p0 + '/spect_' + name + '.png')
            imageio.imwrite(p1 + f'/Mine_{i}.png', fi)

            si1 = (test_mri_tensor[i, 0, :, :]).detach().cpu().numpy()
            si2 = (test_pet_tensor[i, 0, :, :]).detach().cpu().numpy()
            VIFF, SSIM, CC, MSE = Evaluator.VIFF(fi, si1, si2), Evaluator.SSIM(fi, si1, si2), Evaluator.CC(fi, si1,
                                                                                                           si2), Evaluator.MSE(
                fi, si1, si2)
            metric_result += np.array([VIFF, SSIM, CC, MSE])
            save_metrics.write(str(i) + ' ' + str(VIFF) + ' ' + str(SSIM) + ' ' + str(CC) + ' ' + str(MSE) + '\n')

        metric_result /= test_mri_tensor.shape[0]
        print("\t\t VIF\tSSIM\tCC\tMSE")
        print('4Metircs' + '\t' + str(np.round(metric_result[0], 4)) + '\t'
              + str(np.round(metric_result[1], 4)) + '\t'
              + str(np.round(metric_result[2], 4)) + '\t'
              + str(np.round(metric_result[3], 4)))

        save_metrics.write('\n\n' + '4MetircsAvg74MR-SPECT' + ' ' + str(np.round(metric_result[0], 4)) + ' ' + str(
            np.round(metric_result[1], 4)) + ' ' + str(np.round(metric_result[2], 4)) + ' ' + str(
            np.round(metric_result[3], 4)) + '\n')
    elif DataSet=='MRI-PET':
        metric_result = np.zeros((4))
        save_metrics = open(savePath+'/4metrics_42MR-PET.txt', 'a')
        save_metrics.write('i' + ' ' + 'VIFF' + ' ' + 'SSIM' + ' ' + 'CC' + ' ' + 'MSE' + '\n')
        for i in range(test_mri_tensor.shape[0]):
            fi = fused_numpy[i, 0, :, :]
            name = str(i)
            newPath = savePath+'/MR-PET_42/'
            p0 = newPath + 'originalIms/' + name
            p1 = newPath + 'fusedIms/' + name
            if not os.path.exists(p0):
                os.makedirs(p0)
            if not os.path.exists(p1):
                os.makedirs(p1)

            save_image(test_mri_tensor[i, 0, :, :], p0 + '/mri_' + name + '.png')
            save_image(test_pet_tensor[i, 0, :, :], p0 + '/pet_' + name + '.png')
            imageio.imwrite(p1 + f'/Mine_{i}.png', fi)

            si1 = (test_mri_tensor[i, 0, :, :]).detach().cpu().numpy()
            si2 = (test_pet_tensor[i, 0, :, :]).detach().cpu().numpy()
            VIFF, SSIM, CC, MSE = Evaluator.VIFF(fi, si1, si2), Evaluator.SSIM(fi, si1, si2), Evaluator.CC(fi, si1,
                                                                                                           si2), Evaluator.MSE(
                fi, si1, si2)
            metric_result += np.array([VIFF, SSIM, CC, MSE])
            save_metrics.write(str(i) + ' ' + str(VIFF) + ' ' + str(SSIM) + ' ' + str(CC) + ' ' + str(MSE) + '\n')

        metric_result /= test_mri_tensor.shape[0]
        print("\t\t VIF\tSSIM\tCC\tMSE")
        print('4Metircs' + '\t' + str(np.round(metric_result[0], 4)) + '\t'
              + str(np.round(metric_result[1], 4)) + '\t'
              + str(np.round(metric_result[2], 4)) + '\t'
              + str(np.round(metric_result[3], 4)))

        save_metrics.write('\n\n' + 'Avg42MR-PET' + ' ' + str(np.round(metric_result[0], 4)) + ' ' + str(
            np.round(metric_result[1], 4)) + ' ' + str(np.round(metric_result[2], 4)) + ' ' + str(
            np.round(metric_result[3], 4)) + '\n')
    else:
        metric_result = np.zeros((4))
        save_metrics = open(savePath+'/4metrics_21MR-CT.txt', 'a')
        save_metrics.write('i' + ' ' + 'VIFF' + ' ' + 'SSIM' + ' ' + 'CC' + ' ' + 'MSE' + '\n')

        for i in range(test_mri_tensor.shape[0]):
            fi = fused_numpy[i, 0, :, :]
            name = str(i)
            newPath = savePath+'/MR-CT_21/'
            p0 = newPath + 'originalIms/' + name
            p1 = newPath + 'fusedIms/' + name
            if not os.path.exists(p0):
                os.makedirs(p0)
            if not os.path.exists(p1):
                os.makedirs(p1)

            save_image(test_mri_tensor[i, 0, :, :], p0 + '/mri_' + name + '.png')
            save_image(test_pet_tensor[i, 0, :, :], p0 + '/ct_' + name + '.png')
            imageio.imwrite(p1 + f'/Mine_{i}.png', fi)

            si1 = (test_mri_tensor[i, 0, :, :]).detach().cpu().numpy()
            si2 = (test_pet_tensor[i, 0, :, :]).detach().cpu().numpy()
            VIFF, SSIM, CC, MSE = Evaluator.VIFF(fi, si1, si2), Evaluator.SSIM(fi, si1, si2), Evaluator.CC(fi, si1,
                                                                                                           si2), Evaluator.MSE(
                fi, si1, si2)
            metric_result += np.array([VIFF, SSIM, CC, MSE])
            save_metrics.write(str(i) + ' ' + str(VIFF) + ' ' + str(SSIM) + ' ' + str(CC) + ' ' + str(MSE) + '\n')

        metric_result /= test_mri_tensor.shape[0]
        print("\t\t VIF\tSSIM\tCC\tMSE")
        print('4Metircs' + '\t' + str(np.round(metric_result[0], 4)) + '\t'
              + str(np.round(metric_result[1], 4)) + '\t'
              + str(np.round(metric_result[2], 4)) + '\t'
              + str(np.round(metric_result[3], 4)))

        save_metrics.write('\n\n' + 'Avg21MR-CT' + ' ' + str(np.round(metric_result[0], 4)) + ' ' + str(
            np.round(metric_result[1], 4)) + ' ' + str(np.round(metric_result[2], 4)) + ' ' + str(
            np.round(metric_result[3], 4)) + '\n')

    print('Bye!')

