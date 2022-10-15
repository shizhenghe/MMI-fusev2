# -*- coding:utf-8 -*-
#@Project: MMI-Fuse for image fusion
#@Author: Shi Zhenghe, XUST
#@Email: 2571834079@qq.com
#@File : test.py
import datetime
import os
import torch
from torch.autograd import Variable
from net import NestFuse_autoencoder
import utils
from args_fusion import args
import numpy as np


def load_model(path, deepsupervision):
	input_nc = 1
	output_nc = 1
	nb_filter = [64, 112, 160, 208, 256]

	nest_model = NestFuse_autoencoder(nb_filter, input_nc, output_nc, deepsupervision)
	nest_model.load_state_dict(torch.load(path))

	para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
	type_size = 4
	print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * type_size / 1000 / 1000))

	nest_model.eval()
	nest_model.cuda()

	return nest_model


def run_demo(nest_model, infrared_path, visible_path, output_path_root, index, f_type):
	print(infrared_path, '\n', visible_path)
	img_ir, h, w, c = utils.get_test_image(infrared_path)
	img_vi, h, w, c = utils.get_test_image(visible_path)

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
	output_count = 0
	for img_fusion in img_fusion_list:
		file_name =  str(index) +  '_' + 'TA' + '.png'
		output_path = output_path_root + file_name
		# output_path = '/media/qindashuai/KINGSTON/Stone/fusion.png'
		output_count += 1
		# save images
		utils.save_image_test(img_fusion, output_path)
		print(output_path)


def main():
	# run demo

	test_path = "images/IV_images/"
	test_path1 = "outputs/MR_Gad_T2/"
	deepsupervision = False  # true for deeply supervision
	# 'f_information_channel',
	# 				   'f_information_spatial', 'f_criss_cross_channel', 'f_criss_cross_spatial', 'f_channel_spatial',
	# 				   'f_inforamtion_criss_channel', 'f_information_criss_saptial', 'f_information_channel_spatial', 'f_criss_channel_spatail',
	# 				   'f_information_criss_channel_spatial

	fusion_type = ['MR_T1_FDG']

	with torch.no_grad():
		if deepsupervision:
			model_path = args.model_deepsuper
		else:
			model_path = args.model_default
		model = load_model(model_path, deepsupervision)
		print(model)
		for j in range(1):
			output_path = './outputs/' + fusion_type[j]

			if os.path.exists(output_path) is False:
				os.makedirs(output_path)
			output_path = output_path + '/'

			f_type = fusion_type[j]
			print('Processing......  ' + f_type)

			for i in range(10):
				index = i+1
				infrared_path = test_path  + str(index) + '_MR_T10' + '.png'
				visible_path = test_path + 'DEC_Y' + str(index) + '_FDG0' + '.png'
				# infrared_path = '/home/nvidia/fish/imagefusion-nestfuse-master_gai/MMI-fuse/images/IV_images/DEC_Y1_FDG0.png'
				# visible_path = '/home/nvidia/fish/meidcal_image_fusion/MR_FDG_png/norm_brain/MR_T2/1_MR_T20.png'
				curr_time = datetime.datetime.now()
				run_demo(model, infrared_path, visible_path, output_path, index, 'a')
				curr_time1 = datetime.datetime.now()
				print(curr_time1 - curr_time)
	print('Done......')


if __name__ == '__main__':
	main()
