import colorsys
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageSequence
import torch
from torch.autograd import Variable
from net import NestFuse_autoencoder
import utils
from args_fusion import args

import os
import test

save_root = '/home/nvidia/fish/medical_image_fusion/MR_FDG_HSV_DEC/norm_brain/PET/'
read_root = '/home/nvidia/fish/MR_FDG_png'

def run_demo(nest_model, infrared_img, visible_path, f_type):
    img_ir = infrared_img
    img_ir = torch.from_numpy(img_ir).float()
    img_vi, h, w, c = utils.get_test_image(visible_path)
    h, w = img_ir.shape

    # dim = img_ir.shape
    if c is 1:
        if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()
            img_ir = Variable(img_ir, requires_grad=False)
            img_vi = Variable(img_vi, requires_grad=False)
            # encoder
            en_r = nest_model.encoder(img_ir)
            en_v = nest_model.encoder(img_vi)
            # fusion
            f = nest_model.fusion(en_r, en_v, f_type)
            # decoder
            img_fusion_list = nest_model.decoder_eval(f)
    else:
        # fusion each block
        img_fusion_blocks = []
        for i in range(c):
            # encoder
            img_vi_temp = img_vi[i]
            img_ir_temp = img_ir[i]
            if args.cuda:
                img_vi_temp = img_vi_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
            img_vi_temp = Variable(img_vi_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)

            en_r = nest_model.encoder(img_ir_temp)
            en_v = nest_model.encoder(img_vi_temp)
            # fusion
            f = nest_model.fusion(en_r, en_v, f_type)
            # decoder
            img_fusion_temp = nest_model.decoder_eval(f)
            img_fusion_blocks.append(img_fusion_temp)
        img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

            ############################ multi outputs ##############################################

    for img_fusion in img_fusion_list:
        return img_fusion

def HSV_to_RGB(h, s, v):
    rows, cols = h.shape
    h, s, v = np.array(h), np.array(s), np.array(v)

    r, g, b = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))
    for i in range(0, rows):
        for j in range(0, cols):
            r[i][j], g[i][j], b[i][j] = colorsys.hsv_to_rgb(h[i][j], s[i][j], v[i][j])

    new_img = cv.merge((b, g, r))

    return new_img

def RGB_to_HSV(path):
    img = cv.imread(path, 1)
    basename = os.path.basename(path)
    img = Image.fromarray(img.astype('uint8')).convert('RGB')

    b, g, r = img.split()

    b, g, r = np.array(b), np.array(g), np.array(r)
    rows, cols = b.shape

    h, s, v = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))

    for i in range(0, rows):
        for j in range(0, cols):
            h[i][j], s[i][j], v[i][j] = colorsys.rgb_to_hsv(r[i][j], g[i][j], b[i][j])

    v = v
    h = h * 255.0
    s = s * 255.0

    # save!
    # tmp_dir_h = os.path.join(save_root, 'DEC_h/')
    # tmp_dir_s = os.path.join(save_root, 'DEC_s/')
    # tmp_dir_v = os.path.join(save_root, 'DEC_v/')
    #
    # if os.path.exists(tmp_dir_h) is False:
    #     os.makedirs(tmp_dir_h)
    # if os.path.exists(tmp_dir_s) is False:
    #     os.makedirs(tmp_dir_s)
    # if os.path.exists(tmp_dir_v) is False:
    #     os.makedirs(tmp_dir_v)
    #
    # save_h = os.path.join(tmp_dir_h, basename)
    # save_s = os.path.join(tmp_dir_s, basename)
    # save_v = os.path.join(tmp_dir_v, basename)
    #
    # cv.imwrite(save_s, s)
    # cv.imwrite(save_h, h)
    # cv.imwrite(save_v, v)
    #
    # print('saved!')

    return h, s, v

def get_v_img():
    norm_brain_path = '/home/nvidia/fish/medical_image_fusion/MR_FDG_HSV_DEC/norm_brain/PET/DEC_v'

    path = []


    for _, dirs, files in os.walk(norm_brain_path):
        for file in files:
            readpath = norm_brain_path + '/' + file
            path.append(readpath)

    path.sort()

    return path

def get_hs_img():
    norm_brain_path = '/home/nvidia/fish/medical_image_fusion/MR_FDG_HSV_DEC/norm_brain/PET'
    s_path = []
    h_path = []

    for _, dirs, _ in os.walk(norm_brain_path):
        for dir in dirs:
            path = norm_brain_path + '/' + dir
            for _, _, files in os.walk(path):
                for file in files:
                    readpath = path + '/' + file
                    if dir == 'DEC_s':
                        s_path.append(readpath)
                    elif dir == 'DEC_h':
                        h_path.append(readpath)
                    else:
                        continue

    s_path.sort()
    h_path.sort()

    return h_path, s_path

if __name__ == '__main__':
    root_path = '/home/nvidia/fish/medical_image_fusion/MR_FDG_png/norm_brain/PET'
    test_path = "images/IV_images/"
    paths = []
    for _, dirs, files in os.walk(root_path):
        for file in files:
            path = root_path + '/' + file
            paths.append(path)
    paths.sort()

    for i in range(len(paths)):
        h, s, v = RGB_to_HSV(paths[i])
        index = i + 1

        deepsupervision = False  # true for deeply supervision


        with torch.no_grad():
            if deepsupervision:
                model_path = args.model_deepsuper
            else:
                model_path = args.model_default
            model = test.load_model(model_path, deepsupervision)

            print('Processing......  ')
            visible_path = test_path + str(index) + '_MR_T20' + '.png'

            fuse_img = run_demo(model, v, visible_path, 'hsv')

            img = HSV_to_RGB(h, s, fuse_img)
            save_path = '/home/nvidia/fish/medical_image_fusion/My_Test/HSV/MR_T2_PET/' + '{}.png'.format(str(i+1))
            cv.imwrite(save_path, img)
            print('saved!')

    print('Done......')

