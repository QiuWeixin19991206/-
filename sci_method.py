import cv2
import torch.nn as nn
import os
import sys
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
import torch.utils.data
import random
from glob import glob
import torchvision.transforms as transforms
import time

'''数据处理 相当于TensorDataset'''
class MemoryFriendlyLoader(torch.utils.data.Dataset):
    def __init__(self, img_dir, task):

        self.batch_w = 600
        self.batch_h = 400

        # 初始化函数，设置图像目录和任务类型
        self.low_img_dir = img_dir
        self.task = task
        self.train_low_data_names = []

        # 获取图像目录下所有图像文件的路径
        for root, dirs, names in os.walk(self.low_img_dir):
            for name in names:
                self.train_low_data_names.append(os.path.join(root, name))

        # 将图像文件路径排序
        self.train_low_data_names.sort()
        self.count = len(self.train_low_data_names)

        # 定义图像预处理的转换操作
        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def load_images_transform(self, file):
        # 加载图像并进行转换
        im = Image.open(file).convert('RGB')
        img_norm = self.transform(im).numpy()
        img_norm = np.transpose(img_norm, (1, 2, 0))
        return img_norm

    def __getitem__(self, index):
        # 获取索引对应的图像数据和文件名

        # 加载图像并进行预处理
        low = self.load_images_transform(self.train_low_data_names[index])

        h = low.shape[0]
        w = low.shape[1]
        #
        h_offset = random.randint(0, max(0, h - self.batch_h - 1))
        w_offset = random.randint(0, max(0, w - self.batch_w - 1))
        #
        # if self.task != 'test':
        #     low = low[h_offset:h_offset + batch_h, w_offset:w_offset + batch_w]

        # 将图像转换为numpy数组，并转换通道顺序
        low = np.asarray(low, dtype=np.float32)
        low = np.transpose(low[:, :, :], (2, 0, 1))

        img_name = self.train_low_data_names[index].split('\\')[-1]
        # if self.task == 'test':
        #     # img_name = self.train_low_data_names[index].split('\\')[-1]
        #     return torch.from_numpy(low), img_name

        # 返回图像数据和文件名
        return torch.from_numpy(low), img_name

    def __len__(self):
        # 返回数据集的长度
        return self.count

'''loss'''
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.smooth_loss = SmoothLoss()

    def forward(self, input, illu):
        Fidelity_Loss = self.l2_loss(illu, input)
        Smooth_Loss = self.smooth_loss(input, illu)
        return 1.5*Fidelity_Loss + Smooth_Loss

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    # output: output      input:input
    def forward(self, input, output):
        self.output = output
        self.input = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        w1 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :] - self.input[:, :, :-1, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w2 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :] - self.input[:, :, 1:, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w3 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 1:] - self.input[:, :, :, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w4 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-1] - self.input[:, :, :, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w5 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-1] - self.input[:, :, 1:, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w6 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 1:] - self.input[:, :, :-1, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w7 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-1] - self.input[:, :, :-1, 1:], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w8 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 1:] - self.input[:, :, 1:, :-1], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w9 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :] - self.input[:, :, :-2, :], 2), dim=1,
                                 keepdim=True) * sigma_color)
        w10 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :] - self.input[:, :, 2:, :], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w11 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, 2:] - self.input[:, :, :, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w12 = torch.exp(torch.sum(torch.pow(self.input[:, :, :, :-2] - self.input[:, :, :, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w13 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-1] - self.input[:, :, 2:, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w14 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 1:] - self.input[:, :, :-2, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w15 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-1] - self.input[:, :, :-2, 1:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w16 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 1:] - self.input[:, :, 2:, :-1], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w17 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, :-2] - self.input[:, :, 1:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w18 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, 2:] - self.input[:, :, :-1, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w19 = torch.exp(torch.sum(torch.pow(self.input[:, :, 1:, :-2] - self.input[:, :, :-1, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w20 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-1, 2:] - self.input[:, :, 1:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w21 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, :-2] - self.input[:, :, 2:, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w22 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, 2:] - self.input[:, :, :-2, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w23 = torch.exp(torch.sum(torch.pow(self.input[:, :, 2:, :-2] - self.input[:, :, :-2, 2:], 2), dim=1,
                                  keepdim=True) * sigma_color)
        w24 = torch.exp(torch.sum(torch.pow(self.input[:, :, :-2, 2:] - self.input[:, :, 2:, :-2], 2), dim=1,
                                  keepdim=True) * sigma_color)
        p = 1.0

        pixel_grad1 = w1 * torch.norm((self.output[:, :, 1:, :] - self.output[:, :, :-1, :]), p, dim=1, keepdim=True)
        pixel_grad2 = w2 * torch.norm((self.output[:, :, :-1, :] - self.output[:, :, 1:, :]), p, dim=1, keepdim=True)
        pixel_grad3 = w3 * torch.norm((self.output[:, :, :, 1:] - self.output[:, :, :, :-1]), p, dim=1, keepdim=True)
        pixel_grad4 = w4 * torch.norm((self.output[:, :, :, :-1] - self.output[:, :, :, 1:]), p, dim=1, keepdim=True)
        pixel_grad5 = w5 * torch.norm((self.output[:, :, :-1, :-1] - self.output[:, :, 1:, 1:]), p, dim=1, keepdim=True)
        pixel_grad6 = w6 * torch.norm((self.output[:, :, 1:, 1:] - self.output[:, :, :-1, :-1]), p, dim=1, keepdim=True)
        pixel_grad7 = w7 * torch.norm((self.output[:, :, 1:, :-1] - self.output[:, :, :-1, 1:]), p, dim=1, keepdim=True)
        pixel_grad8 = w8 * torch.norm((self.output[:, :, :-1, 1:] - self.output[:, :, 1:, :-1]), p, dim=1, keepdim=True)
        pixel_grad9 = w9 * torch.norm((self.output[:, :, 2:, :] - self.output[:, :, :-2, :]), p, dim=1, keepdim=True)
        pixel_grad10 = w10 * torch.norm((self.output[:, :, :-2, :] - self.output[:, :, 2:, :]), p, dim=1, keepdim=True)
        pixel_grad11 = w11 * torch.norm((self.output[:, :, :, 2:] - self.output[:, :, :, :-2]), p, dim=1, keepdim=True)
        pixel_grad12 = w12 * torch.norm((self.output[:, :, :, :-2] - self.output[:, :, :, 2:]), p, dim=1, keepdim=True)
        pixel_grad13 = w13 * torch.norm((self.output[:, :, :-2, :-1] - self.output[:, :, 2:, 1:]), p, dim=1, keepdim=True)
        pixel_grad14 = w14 * torch.norm((self.output[:, :, 2:, 1:] - self.output[:, :, :-2, :-1]), p, dim=1, keepdim=True)
        pixel_grad15 = w15 * torch.norm((self.output[:, :, 2:, :-1] - self.output[:, :, :-2, 1:]), p, dim=1, keepdim=True)
        pixel_grad16 = w16 * torch.norm((self.output[:, :, :-2, 1:] - self.output[:, :, 2:, :-1]), p, dim=1, keepdim=True)
        pixel_grad17 = w17 * torch.norm((self.output[:, :, :-1, :-2] - self.output[:, :, 1:, 2:]), p, dim=1, keepdim=True)
        pixel_grad18 = w18 * torch.norm((self.output[:, :, 1:, 2:] - self.output[:, :, :-1, :-2]), p, dim=1, keepdim=True)
        pixel_grad19 = w19 * torch.norm((self.output[:, :, 1:, :-2] - self.output[:, :, :-1, 2:]), p, dim=1, keepdim=True)
        pixel_grad20 = w20 * torch.norm((self.output[:, :, :-1, 2:] - self.output[:, :, 1:, :-2]), p, dim=1, keepdim=True)
        pixel_grad21 = w21 * torch.norm((self.output[:, :, :-2, :-2] - self.output[:, :, 2:, 2:]), p, dim=1, keepdim=True)
        pixel_grad22 = w22 * torch.norm((self.output[:, :, 2:, 2:] - self.output[:, :, :-2, :-2]), p, dim=1, keepdim=True)
        pixel_grad23 = w23 * torch.norm((self.output[:, :, 2:, :-2] - self.output[:, :, :-2, 2:]), p, dim=1, keepdim=True)
        pixel_grad24 = w24 * torch.norm((self.output[:, :, :-2, 2:] - self.output[:, :, 2:, :-2]), p, dim=1, keepdim=True)

        ReguTerm1 = torch.mean(pixel_grad1) \
                    + torch.mean(pixel_grad2) \
                    + torch.mean(pixel_grad3) \
                    + torch.mean(pixel_grad4) \
                    + torch.mean(pixel_grad5) \
                    + torch.mean(pixel_grad6) \
                    + torch.mean(pixel_grad7) \
                    + torch.mean(pixel_grad8) \
                    + torch.mean(pixel_grad9) \
                    + torch.mean(pixel_grad10) \
                    + torch.mean(pixel_grad11) \
                    + torch.mean(pixel_grad12) \
                    + torch.mean(pixel_grad13) \
                    + torch.mean(pixel_grad14) \
                    + torch.mean(pixel_grad15) \
                    + torch.mean(pixel_grad16) \
                    + torch.mean(pixel_grad17) \
                    + torch.mean(pixel_grad18) \
                    + torch.mean(pixel_grad19) \
                    + torch.mean(pixel_grad20) \
                    + torch.mean(pixel_grad21) \
                    + torch.mean(pixel_grad22) \
                    + torch.mean(pixel_grad23) \
                    + torch.mean(pixel_grad24)
        total_term = ReguTerm1
        return total_term

'''网络模型'''
class EnhanceNetwork(nn.Module):
    def __init__(self, layers, channels):
        # 调用父类的构造函数，完成初始化
        super(EnhanceNetwork, self).__init__()
        # 卷积核大小为3*3、膨胀设为1、通过计算，得到合适的填充大小。
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        # 输入卷积层
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.ReLU()
        )
        # 中间卷积层
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        # 模块列表
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.conv)
        # 输出卷积层
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # 将输入数据通过输入卷积层 self.in_conv，得到中间特征 fea。
        fea = self.in_conv(input)
        # 通过多个中间卷积层 self.blocks 进行迭代，
        for conv in self.blocks:
            # 每次迭代中将当前特征 fea 与中间卷积层的输出相加。
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        # 将输出特征与输入数据相加，得到增强后的图像。
        illu = fea + input
        # 通过 torch.clamp 函数将图像的像素值限制在 0.0001 到 1 之间，以确保输出在有效范围内。
        illu = torch.clamp(illu, 0.0001, 1)

        return illu


class CalibrateNetwork(nn.Module):
    def __init__(self, layers, channels):
        super(CalibrateNetwork, self).__init__()
        kernel_size = 3
        dilation = 1
        padding = int((kernel_size - 1) / 2) * dilation
        self.layers = layers

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.blocks = nn.ModuleList()
        for i in range(layers):
            self.blocks.append(self.convs)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        # 将输入数据通过输入卷积层，得到中间特征 fea。
        fea = self.in_conv(input)
        for conv in self.blocks:
            fea = fea + conv(fea)

        fea = self.out_conv(fea)
        # 计算输入与输出的差异，得到增益调整的值。
        delta = input - fea

        return delta
class Finetunemodel(nn.Module):

    def __init__(self, weights):
        super(Finetunemodel, self).__init__()
        self.enhance = EnhanceNetwork(layers=1, channels=3)
        self._criterion = LossFunction()

        base_weights = torch.load(weights)
        pretrained_dict = base_weights
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()

        if isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1., 0.02)

    def forward(self, input):
        i = self.enhance(input)
        r = input / i
        r = torch.clamp(r, 0, 1)
        return i, r


    def _loss(self, input):
        i, r = self(input)
        loss = self._criterion(input, i)
        return loss

class sci_method:
    #定义了一个函数 save_images()，用于将张量保存为图像文件。这个函数将张量转换为 NumPy 数组，并使用 PIL 库将数组保存为图像文件。
    def save_images(self, tensor, path):
        image_numpy = tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
        im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
        im.save(path, 'png')

    #判断哪些图片需要处理，threshold是亮度阈值
    def is_dark(self, image_path, threshold=70):
        img = cv2.imread(image_path, 0)#读取灰度图
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#转灰度图
        averge = cv2.mean(img)[0]
        if averge < threshold:
            print(image_path, averge)
        return averge < threshold #1是低照度，0不是
    #从data_path中，保存低照图片到train_path
    def process_dark(self, data_path, train_path):
        data_path = glob(os.path.join(data_path, '*.[pj][pn]g'))#递归获取目录下的所有png\jpg文件路径
        os.makedirs(train_path, exist_ok=True)
        for filename in data_path:#遍历输入目录的文件
            #读取png和jpg
            # if os.path.isfile(filename) and filename.lower().endswith(('.png', 'jpg', 'jpeg')):
            if self.is_dark(filename):
                image_name = filename.split('/')[-1].split('.')[0]  # 处理图片名称，将其从完整的文件路径中提取出来，只保留文件名而去掉路径和扩展名
                u_name = '%s.png' % (image_name)  # 构造输出图片的文件名，这里使用原始图片的文件名，并指定保存为 .png 格式
                u_path = train_path + '/' + u_name
                img = cv2.imread(filename)
                cv2.imwrite(f'{u_path}', img)

        print('save low-light to train_path!')
        return None

    def run(self, in_path):
        strat_time = time.time()
        parser = argparse.ArgumentParser("SCI")#创建了一个命令行解析器对象
        #定义命令行参数
        parser.add_argument('--data_path', type=str, default='./train', help='location of the data corpus')#需要光照加强的图片路径
        parser.add_argument('--save_path', type=str, default='./results', help='location of the data corpus')#处理后的图片保存路径
        parser.add_argument('--model', type=str, default='weights.pt', help='location of the data corpus')#模型pt路径
        parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
        parser.add_argument('--seed', type=int, default=2, help='random seed')

        '''创建保存目录'''
        args = parser.parse_args()#解析命令行参数，并将解析结果存储在变量args中
        save_path = args.save_path#通过args.save_path来获取命令行中传入的--save_path参数的值
        os.makedirs(save_path, exist_ok=True)
        data_path = args.data_path  # 通过args.save_path来获取命令行中传入的--save_path参数的值
        os.makedirs(data_path, exist_ok=True)

        self.process_dark(in_path, data_path)#保存需要处理的图片

        TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')#数据处理 相当于TensorDataset
        test_queue = torch.utils.data.DataLoader(TestDataset, batch_size=1, pin_memory=True, num_workers=0)

        if not torch.cuda.is_available():
            print('no gpu device available')
            sys.exit(1)#状态码为1表示程序以异常的方式退出

        model = Finetunemodel(args.model)#通过 args.model 获取了模型.pt的路径，加载该模型
        model = model.cuda()

        model.eval()
        with torch.no_grad():
            for j, (input, image_name) in enumerate(test_queue):

                # input = Variable(input, volatile=True).cuda()#版本太老了，替代为下面这句
                input = input.cuda()
                image_name = image_name[0].split('/')[-1].split('.')[0]#处理图片名称，将其从完整的文件路径中提取出来，只保留文件名而去掉路径和扩展名
                i, r = model(input)
                u_name = '%s.png' % (image_name)#构造输出图片的文件名，这里使用原始图片的文件名，并指定保存为 .png 格式
                print('processing {}'.format(u_name))
                u_path = save_path + '/' + u_name
                self.save_images(r, u_path)
        print('spead time: %.2f' % (time.time() - strat_time))
        return None

if __name__ == '__main__':
    ''' 
        从data中识别低光照图片 
        并保存至train 
        增强后保存至results
    '''

    data_path = './data'
    sci = sci_method()
    sci.run(in_path=data_path)


















