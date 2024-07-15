import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LDRToTRF(nn.Module):
    def __init__(self, **kwargs):
        super(LDRToTRF, self ).__init__(**kwargs)
        
    def forward(self, d1layer):
        batchSize = d1layer.size()[0]
        
        d1lext = torch.zeros(batchSize, 256)
        d1lext[:, 1:] = d1layer.squeeze(1)

        tf = torch.cumsum(d1lext, dim=1)

        tf = tf * 2 - 1        
       
        return tf # batch size transformation function

class intensityTransform(nn.Module):
    def __init__(self, intensities, channels, **kwargs):
        super(intensityTransform, self).__init__(**kwargs)
        self.channels = channels
        self.scale = intensities - 1

    def get_config(self):
        config = super(intensityTransform, self).get_config()
        config.update({'channels': self.channels, 'scale': self.scale})
        return config

    def forward(self, inputs):
        images, transforms = inputs

        if self.channels == 1:
            transforms = transforms.unsqueeze(1).unsqueeze(3) 
            #transforms = transforms.unsqueeze(3) 
        else:
            transforms = transforms.unsqueeze(3)

        images = 0.5 * images + 0.5
        images = torch.round(self.scale * images)
        images = images.type(torch.LongTensor)
        images = images.to(device)
        transforms = transforms.to(device)
        minimum_w = images.size(3)
        iter_n = 0
        temp = 1
        while minimum_w > temp:
            temp *= 2
            iter_n += 1

        images = torch.split(images, 1, dim=1)

        if self.channels == 1:
            for i in range(iter_n):
                transforms = torch.cat([transforms, transforms], dim=3)

            r = torch.gather(input=transforms, dim=2, index=images[0])
            g = torch.gather(input=transforms, dim=2, index=images[1])
            b = torch.gather(input=transforms, dim=2, index=images[2])
        else:
            for i in range(iter_n):
                transforms = torch.cat([transforms, transforms], dim=3)

            transforms = torch.split(transforms, 1, dim=1)

            r = torch.gather(input=transforms[0], dim=2, index=images[0])
            g = torch.gather(input=transforms[1], dim=2, index=images[1])
            b = torch.gather(input=transforms[2], dim=2, index=images[2])

        xx = torch.cat([r, g, b], dim=1)

        return xx

class invertedResidual(nn.Module):
    def __init__(self, input_ch=32, output_ch=16, kernel_size=3, strides=1,
                 expand_ratio=6, se_ratio=0.25, dropout_rate=0.1):
        super(invertedResidual, self).__init__()
        padding = kernel_size//2
        self.se_ratio = se_ratio
        self.dropout_rate = dropout_rate
        outputs_e = input_ch * expand_ratio
        self.ir_conv1 = nn.Conv2d(input_ch, outputs_e, kernel_size=1, stride=1, bias=False)
        self.ir_batchNorm1 = nn.BatchNorm2d(outputs_e)
        self.ir_depthwise = nn.Conv2d(outputs_e, outputs_e, kernel_size=kernel_size, stride=strides, padding=padding,
                                      groups=outputs_e, bias=False)
        self.ir_batchNorm2 = nn.BatchNorm2d(outputs_e)

        if 0 < se_ratio <= 1:
            outputs_se = max(1, int(input_ch * se_ratio))
            self.se_globalAvgPool = nn.AdaptiveAvgPool2d(1)

            self.se_conv1 = nn.Conv2d(outputs_e, outputs_se, kernel_size=1, stride=1, bias=False)
            self.se_conv2 = nn.Conv2d(outputs_se, outputs_e, kernel_size=1, stride=1, bias=False)

        self.ir_conv2 = nn.Conv2d(outputs_e, output_ch, kernel_size=1, stride=1, bias=False)
        self.ir_batchNorm3 = nn.BatchNorm2d(output_ch)

        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.ir_conv1(x)
        x = self.ir_batchNorm1(x)
        x = self.swish(x)
        x = self.ir_depthwise(x)
        x = self.ir_batchNorm2(x)
        x = self.swish(x)
        if 0 < self.se_ratio <= 1:
            s = self.se_globalAvgPool(x)
            s = self.se_conv1(s)
            s = self.swish(s)
            s = self.se_conv2(s)
            s = torch.sigmoid(s)
            x = torch.mul(x, s)
        x = self.ir_conv2(x)
        x = self.ir_batchNorm3(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x


class convBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, strides, dropout_rate=0.1):
        super(convBlock, self).__init__()
        self.dropout_rate = dropout_rate
        padding = kernel_size//2
        self.cb_conv1 = nn.Conv2d(input_ch, output_ch, kernel_size, strides, padding=padding, bias=False)
        self.cb_batchNorm = nn.BatchNorm2d(output_ch)
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.cb_conv1(x)
        x = self.cb_batchNorm(x)
        x = self.swish(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x

class localConvBlock(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, strides, dropout_rate=0.1):
        super(localConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        padding = kernel_size//2
        self.lcb_conv1 = nn.Conv2d(input_ch, output_ch, kernel_size, strides, padding=padding, bias=False)
        self.lcb_batchNorm = nn.BatchNorm2d(output_ch)
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def swish(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = self.lcb_conv1(x)
        x = self.lcb_batchNorm(x)
        x = self.swish(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x

class LDRMNet(nn.Module):    
    def __init__(self):
        super(LDRMNet, self).__init__()

        self.global_stage1 = convBlock(3, 16, 5, 2, dropout_rate=0.1)
        self.global_stage2 = invertedResidual(16, 24, 5, 2, dropout_rate=0.1)
        self.global_stage3 = invertedResidual(24, 40, 5, 2, dropout_rate=0.1)
        self.global_stage4 = invertedResidual(40, 80, 5, 2, dropout_rate=0.1)

        self.local_stage2 = localConvBlock(16, 24, 3, 2, dropout_rate=0.1)
        self.local_stage3 = localConvBlock(24, 40, 3, 2, dropout_rate=0.1)
        self.local_stage4 = localConvBlock(40, 80, 3, 2, dropout_rate=0.1)

        self.global_stage5 = invertedResidual(160, 112, 5, 2, dropout_rate=0.1)
        #self.global_stage6 = convBlock(112, 255, 1, 1, dropout_rate=0.1) for dl now test for tf
        self.global_stage6 = convBlock(112, 768, 1, 1, dropout_rate=0.1)
        self.global_stage6_globalPool = nn.AdaptiveAvgPool2d(1)

        
        self.FC = nn.Linear(768, 255)
        self.ldrtotf = LDRToTRF()
        self.softmax = nn.Softmax(dim = 2)
        self.intensity_trans = intensityTransform(intensities=256, channels=1)

    def forward(self, x):
        x_256 = F.interpolate(x, 256)

        gx = self.global_stage1(x_256)
        gxx = self.global_stage2(gx)
        gxx = self.global_stage3(gxx)
        gxx = self.global_stage4(gxx)
        lx = self.local_stage2(gx)
        lx = self.local_stage3(lx)
        lx = self.local_stage4(lx)
        fx = torch.cat([gxx, lx], dim=1)
        fx = self.global_stage5(fx)
        fx = self.global_stage6(fx)
        fx = self.global_stage6_globalPool(fx)

        ff = torch.squeeze(fx, dim=2)
        ff = torch.squeeze(ff, dim=2)
        ff = torch.unsqueeze(ff, dim=1)

        ff = self.FC(ff)

        ff = torch.tanh(ff)
        ff = (ff+1)/2
        dl = ff / ff.sum(dim=2, keepdim=True)
        tf = self.ldrtotf(dl)        

        xy = self.intensity_trans((x, tf))

        return xy, dl, tf