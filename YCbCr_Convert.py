import colorsys
import cv2 as cv
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageSequence

import os
save_root = '/home/nvidia/fish/MR_FDG_DEC'
read_root = '/home/nvidia/fish/MR_FDG_png'
def gif_png():
    path = '/home/nvidia/fish/MR_FDG'

    for root, dirs, files in os.walk(path):
        for dir in dirs:
            path1 = root + '/' + dir
            for root1, dirs1, files1 in os.walk(path1):
                for file in files1:
                    readpath = root1 + '/' + file
                    name = file.split('.')[-2]
                    print(readpath)
                    if dir == 'norm_brain':
                        save_path = save_root + '/' + dir
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img = Image.open(readpath)
                        i = 0
                        for frame in ImageSequence.Iterator(img):
                            frame.save(save_path + '/' + name + str(i) + '.png')
                            print('saved!')
                            i += 1
                    elif dir == 'Alzheimer':
                        save_path = save_root + '/' + dir
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img = Image.open(readpath)
                        i = 0
                        for frame in ImageSequence.Iterator(img):
                            frame.save(save_path + '/' + name + str(i) + '.png')
                            print('saved!')
                            i += 1
                    elif dir == 'Huntington':
                        save_path = save_root + '/' + dir
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img = Image.open(readpath)
                        i = 0
                        for frame in ImageSequence.Iterator(img):
                            frame.save(save_path + '/' + name + str(i) + '.png')
                            print('saved!')
                            i += 1
                    elif dir == 'Glioma':
                        save_path = save_root + '/' + dir
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img = Image.open(readpath)
                        i = 0
                        for frame in ImageSequence.Iterator(img):
                            frame.save(save_path + '/' + name + str(i) + '.png')
                            print('saved!')
                            i += 1
                    else:
                        save_path = save_root + '/' + dir
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        img = Image.open(readpath)
                        i = 0
                        for frame in ImageSequence.Iterator(img):
                            frame.save(save_path + '/' + name + str(i) + '.png')
                            print('saved!')
                            i += 1

def get_Y_img():
    norm_brain_path = './outputs/MR_T1_FDG'

    path = []


    for _, dirs, files in os.walk(norm_brain_path):
        for file in files:
            readpath = norm_brain_path + '/' + file
            path.append(readpath)

    path.sort()

    return path

def get_CC_img():
    norm_brain_path = 'G:/Stone/MR_FDG_DEC/norm_brain'
    Cb_path = []
    Cr_path = []

    for _, dirs, _ in os.walk(norm_brain_path):
        for dir in dirs:
            path = norm_brain_path + '/' + dir
            for _, _, files in os.walk(path):
                for file in files:
                    readpath = path + '/' + file
                    if dir == 'DEC_Cr':
                        Cr_path.append(readpath)
                    elif dir == 'DEC_Cb':
                        Cb_path.append(readpath)
                    else:
                        continue

    Cb_path.sort()
    Cr_path.sort()

    return Cb_path, Cr_path

def RGB2YCrCb(path):
    img = Image.open(path)
    img_ycrcb = img.convert('YCbCr')

    Y, Cb, Cr = img_ycrcb.split()

    basename = os.path.basename(path)

    Y.save(save_root + '/' + 'AIDS' + '/' + 'SPECT' + '/' + 'DEC_Y' + '/' + basename)
    Cb.save(save_root + '/' + 'AIDS' + '/' + 'SPECT' + '/' + 'DEC_Cb' + '/' + basename)
    Cr.save(save_root + '/' + 'AIDS' + '/' + 'SPECT' + '/' + 'DEC_Cr' + '/' + basename)

    print('saved!')

    return Y, Cb, Cr

def YCrCb2RGB(Y, Cb, Cr):
    img_ycbcr = Image.merge('YCbCr', (Y, Cb, Cr))

    img = img_ycbcr.convert('RGB')

    return img

def HSV_to_RGB(h, s, v):
	rows, cols = h.shape

	r, g, b = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))
	for i in range(0, rows):
		for j in range(0, cols):
			r[i][j], g[i][j], b[i][j] = colorsys.hsv_to_rgb(h[i][j], s[i][j], v[i][j])

	new_img = cv.merge((b, g, r))

	return new_img

def RGB_to_HSV(img):
	"""
	:param img: 输入的图片是由cv2.imread读取的数据
	:return:
	"""

	img = Image.fromarray(img.astype('uint8')).convert('RGB')

	b, g, r = img.split()

	b, g, r = np.array(b), np.array(g), np.array(r)
	rows, cols = b.shape

	h, s, v = np.zeros((rows, cols)), np.zeros((rows, cols)), np.zeros((rows, cols))

	for i in range(0, rows):
		for j in range(0, cols):
			h[i][j], s[i][j], v[i][j] = colorsys.rgb_to_hsv(r[i][j], g[i][j], b[i][j])

	v = v / 255.0

	return h, s, v

if __name__ == '__main__':
    paths = get_Y_img()
    Cb_path, Cr_path = get_CC_img()
    print(paths, Cr_path, Cb_path)
    for i in range(len(paths)):
        Y = Image.open(paths[i])
        Cr = Image.open(Cr_path[i])
        Cb = Image.open(Cb_path[i])
        img = YCrCb2RGB(Y, Cb, Cr)
        # save_path = '/home/nvidia/fish/meidcal_image_fusion/nestfusion/MR_T1_FDG_fusion' + '/' + str(i + 1) + 'MR_T1_FDG'
        img.save('{}c.png'.format(str(1+i)))
        # img = YCrCb2RGB(Y, Cb, Cr)