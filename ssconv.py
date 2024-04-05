'''
Description:
Date: 2023-07-21 14:36:27
LastEditTime: 2023-07-27 18:41:47
FilePath: /chengdongzhou/ScConv.py
'''
import torch
import torch.nn.functional as F
import torch.nn as nn

"""
对于输入的特征进行归一化处理
"""
class GroupBatchnorm2d(nn.Module):
    def __init__(self,
                 # 特征通道数
                 c_num: int,
                 # 分组数
                 group_num: int = 2,
                 # 归一化过程中防止分母为零的小数
                 eps: float = 1e-10,
                 ):
        super(GroupBatchnorm2d, self).__init__()
        # 确保特征通道数不小于分组数
        assert c_num >= group_num
        self.group_num = group_num
        # 用于归一化后的特征进行缩放
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))
        # 用于归一化后的特征进行偏移
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        # N:batch size    C:通道数   H，W:高度和宽度 16*5*6*9
        N, C, H, W = x.size()
        # 将特征张量x重塑成为(N, group_num, -1)的形状
        x = x.view(N, self.group_num, -1)
        # 在第三个维度上计算分组的平均值
        mean = x.mean(dim=2, keepdim=True)
        # 在第三个维度上计算分组的标准差
        std = x.std(dim=2, keepdim=True)
        # 归一化
        x = (x - mean) / (std + self.eps)
        # 将归一化后的特征张量重塑回原始形状(N, C, H, W)
        x = x.view(N, C, H, W)
        # 归一化后的特征进行缩放和偏移，然后返回结果，这一步缩放和偏移应用于每个通道的特征
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 # 输出通道数
                 oup_channels: int,
                 # 分组数
                 group_num: int = 2,
                 # 门限值
                 gate_treshold: float = 0.5,
                 # bool值，用于显示nn.GroupNorm是否进行分组归一化
                 torch_gn: bool = True
                 ):
        super().__init__()
        # 分组归一化层
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        # 门的阈值
        self.gate_treshold = gate_treshold
        # sigmoid函数
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        # 归一化处理输入特征x
        gn_x = self.gn(x)
        # 归一化后的权重参数计算得到加权参数
        w_gamma = self.gn.weight / sum(self.gn.weight)
        w_gamma = w_gamma.view(1, -1, 1, 1)
        # 门控系数，通过加权特征进行sigmoid函数处理得到
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate 门控系数将输入特征x进行分组
        w1 = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts)  # 大于门限值的设为1，否则保留原值
        w2 = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts)  # 大于门限值的设为0，否则保留原值
        x_1 = w1 * x
        x_2 = w2 * x
        # 通过重构函数 reconstruct 对分组后的特征进行融合得到最终输出
        y = self.reconstruct(x_1, x_2)
        return y

    def reconstruct(self, x_1, x_2):
        # 将门控后的输入特征x_1和x_2进行分组
        # torch.split 函数：按照通道数的一半将特征张量进行拆分
        a=x_1.size(1) // 2
        b=x_2.size(1) // 2
        x_11, x_12 = torch.split(x_1, [a,a+1], dim=1)
        x_21, x_22 = torch.split(x_2, b+1, dim=1)
        # print(x_11.size(1),x_12.size(1),x_21.size(1),x_22.size(1))
        # torch.cat 函数：将拆分后的特征张量按照指定维度进行拼接，形成最终的输出特征
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 # 输出通道数
                 op_channel: int,
                 # 控制通道分配的参数，表示高层特征和低层特征的通道比例
                 alpha: float = 1 / 2,
                 # 压缩比例
                 squeeze_radio: int = 2,
                 # 分组卷积的组数
                 group_size: int = 2,
                 # 分组卷积核大小
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        # 高层通道数和底层通道数
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        # 对高层和底层特征进行压缩的1*1卷积层
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up 高层特征的组卷积层，用于对压缩后的高层特征进行重构
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        # 高层和底层特征的1*1卷积层，用于通道的变换和压缩
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        # 自适应平均池化层，用于在通道维度上进行全局平均池化
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split 将输入特征张量x按照通道分割为高层和低层特征
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        # 分别对高层和底层特征进行压缩
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        # 对压缩后的高层特征进行组卷积和通道变换
        Y1 = self.GWC(up) + self.PWC1(up)
        # 对低层特征进行通道变换
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        # 对变换后的特征按通道维度进行拼接
        out = torch.cat([Y1, Y2], dim=1)
        # 对拼接后的特征进行softmax操作，用于学习通道权重
        out = F.softmax(self.advavg(out), dim=1) * out
        # 将softmax后的特征按照通道分割为两部分
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        # 返回两部分元素级加和
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 # 输出通道数
                 op_channel: int,
                 # SRU模块参数，控制门控特征融合行为
                 group_num: int = 1,
                 gate_treshold: float = 0.5,
                 # CRU模块参数，控制通道重构单元行为
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 1,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x


if __name__ == '__main__':
    x = torch.randn(305, 5, 6, 9)
    model = ScConv(5)
    result=model(x)
    print(model(x).shape)