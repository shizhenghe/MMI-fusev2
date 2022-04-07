import torch
import numpy as np
import cv2 as cv
import torchvision.transforms as transforms
import torch.nn.functional as F
# import sympy.vector

from torch import nn
from torch.nn import Softmax

from torch.autograd import Variable

EPSILON = 1e-5

# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2, fusion_type):
    # avg, max, nuclear
    f_information, inf_deg = information_fusion(tensor1, tensor2)
    f_criss_cross = Criss_Cross_fusion(tensor1, tensor2)
    f_channel = channel_fusion(tensor1, tensor2, p_type='avg')
    f_spatial = spatial_fusion(tensor1, tensor2)

    if fusion_type == 'f_inforamtion_criss_cross':
        tensor_f = f_information
    elif fusion_type == 'f_information_channel':
        tensor_f = torch.sqrt(0.1 * (torch.pow(f_information, 2) + 0.9 * torch.pow(f_channel, 2)) / 2)
    elif fusion_type == "nestfuse":
        tensor_f = (f_channel + f_spatial) / 2
    elif fusion_type == 'f_information_spatial':
        tensor_f = torch.sqrt((torch.pow(f_information, 2) + torch.pow(f_spatial, 2)) / 2)
    elif fusion_type == 'f_criss_cross_channel':
        tensor_f = torch.sqrt((torch.pow(f_criss_cross, 2) + torch.pow(f_channel, 2)) / 2)
    elif fusion_type == 'f_criss_cross_spatial':
        tensor_f = torch.sqrt((torch.pow(f_criss_cross, 2) + torch.pow(f_spatial, 2)) / 2)
    elif fusion_type == 'f_channel_spatial':
        tensor_f = torch.sqrt((torch.pow(f_spatial, 2) + torch.pow(f_channel, 2)) / 2)
    elif fusion_type == 'f_inforamtion_criss_channel':
        tensor_f = torch.sqrt((torch.pow(f_information, 2) + torch.pow(f_channel, 2) + torch.pow(f_criss_cross, 2)) / 3)
    elif fusion_type == 'f_information_criss_saptial':
        tensor_f = torch.sqrt((torch.pow(f_information, 2) + torch.pow(f_spatial, 2) + torch.pow(f_criss_cross, 2)) / 3)
    elif fusion_type == 'f_information_channel_spatial':
        tensor_f = torch.sqrt((torch.pow(f_information, 2) + torch.pow(f_spatial, 2) + torch.pow(f_channel, 2)) / 3)
    elif fusion_type == 'f_criss_channel_spatail':
        tensor_f = torch.sqrt((torch.pow(f_channel, 2) + torch.pow(f_spatial, 2) + torch.pow(f_criss_cross, 2)) / 3)
    elif fusion_type == 'f_information_criss_channel_spatial':
        tensor_f = torch.square((torch.pow(f_information, 2) + torch.pow(f_channel, 2) + torch.pow(f_spatial, 2) + torch.pow(f_criss_cross, 2)) / 4)
    else:
        # tensor_f = inf_deg[:, 0] * tensor1 + inf_deg[:, 1] * tensor2
        tensor_f = f_channel
    return tensor_f

def Criss_Cross_fusion(tensor1, tensor2):
		bs, channel, height, weight = tensor1.size()
		cca = CrissCrossAttention(channel).cuda()
		global_p1 = cca(tensor1)
		global_p2 = cca(tensor2)

		global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
		global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

		tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

		return tensor_f

def information_fusion(tensor1, tensor2):
		fgs1 = computer_grad(tensor1)
		fgs2 = computer_grad(tensor2)

		inf_deg1 = information_preservation_degree(fgs1)
		inf_deg2 = information_preservation_degree(fgs2)

		inf_deg = torch.nn.functional.softmax(torch.cat([torch.unsqueeze(inf_deg1, -1), torch.unsqueeze(inf_deg2, -1)], -1), -1)

		tensor_f = inf_deg[:, 0] * tensor1 + inf_deg[:, 1] * tensor2

		return tensor_f, inf_deg

# select channel
def channel_fusion(tensor1, tensor2, p_type):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# channel attention
def channel_attention(tensor, pooling_type='avg'):
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type is 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type is 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type is 'attention_nuclear':
        pooling_function = nuclear_pooling
    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True) / tensor.size()[1]
    return spatial


# pooling function
def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1)).cuda()

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).cuda()
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).cuda()
        energy_H = (torch.bmm(proj_query_H, proj_key_H).cuda() + self.INF(m_batchsize, height, width).cuda()).view(
            m_batchsize, width, height, height).permute(0, 2, 1, 3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width).cuda()
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                 height).cuda()
        # print(concate)
        # print(att_H)
        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width).cuda()
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3,
                                                                                                             1).cuda()
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1,
                                                                                                             3).cuda()
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x.cuda()


def computer_grad(tensor):
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]]
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = nn.Parameter(data=kernel, requires_grad=False).cuda()

    # kernel = torch.Tensor(kernel)
    # kernel = torch.unsqueeze(kernel, -1)
    # kernel = torch.unsqueeze(kernel, -1).permute((2, 3, 0, 1))

    # kernel = torch.constant([])
    # kernel = tf.expand_dims(kernel, axis=-1)
    # kernel = tf.expand_dims(kernel, axis=-1)

    _, c, _, _ = tensor.shape
    c = int(c)
    for i in range(c):
        fg = F.conv2d(torch.unsqueeze(tensor[:, i, :, :], dim=1), weight=kernel, stride=1, padding=(1, 1))
        if i == 0:
            fgs = fg
        else:
            fgs = torch.cat([fgs, fg], dim=1)

    return fgs


def information_preservation_degree(fgs):
    _, c, _, _ = fgs.shape
    means = 0
    for i in range(0, c):
        mean = torch.mean(torch.square(fgs[:, i, :, :]), dim=[1, 2])
        means += mean
    ws = torch.unsqueeze(means, -1)
    s = torch.mean(ws, -1) / c

    return s

