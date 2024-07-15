import torch
import torchvision
import math

from torchmetrics.functional import structural_similarity_index_measure as SSIM
from torchmetrics.functional import peak_signal_noise_ratio as PSNR

class pixDist(torch.nn.Module):
    def __init__(self, **kwargs):
        super(pixDist, self).__init__(**kwargs)
        
        self.r = torch.arange(256).unsqueeze(1).cuda()
        self.c = torch.arange(256).unsqueeze(0).cuda()
        mask = self.r < self.c
        self.coef = (self.c - self.r) * mask

    def forward(self, y_pred):
        batchSize, channel, row, col = y_pred.size()
        
        batchSum = torch.zeros(batchSize).cuda()

        for b in range(batchSize):
            hist = torch.histc(y_pred[b, ], bins=256)
            sumDist = (hist[self.r] * hist[self.c] * self.coef)
            batchSum[b] = torch.sum(sumDist) / (row*col*(row*col-1))

        return torch.mean(batchSum)      

class DE(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DE, self).__init__(**kwargs)

    def forward(self, y_pred):
        batchSize = y_pred.size()[0]

        value = 0
        for b in range(batchSize):
            hist = torch.histc(y_pred[b, ], bins=256)

            pIn = hist/torch.sum(hist)
            logPin = torch.log2(torch.where(pIn==0, torch.ones_like(pIn), pIn))
            value += torch.sum((-pIn) * (logPin))

        value = value / batchSize

        return value

class EME(torch.nn.Module):
    def __init__(self, winSize=8, **kwargs):
        super(EME, self).__init__(**kwargs)
        self.windowSize = winSize
        self.pooling = torch.nn.MaxPool2d(kernel_size=winSize, stride=winSize, padding=0)

    def forward(self, y_pred):
        batchSize, channel, row, col = y_pred.size()

        maxPred = self.pooling(y_pred)
        minPred = -1 * self.pooling(-1*y_pred)

        batchEME = torch.zeros(batchSize, 1)

        for b in range(batchSize):
            nzIn = torch.nonzero(maxPred[b] - minPred[b], as_tuple=True)
            batchEME[b] = torch.sum(20*torch.log(maxPred[b][nzIn] / (minPred[b][nzIn]+1e-4))) / (row * col) * (self.windowSize*self.windowSize)

        return torch.mean(batchEME)


class evaluationLDR(torch.nn.Module):
    def __init__(self, **kwargs):
        super(evaluationLDR, self).__init__(**kwargs)        
        self.pixDist = pixDist()
        self.DE = DE()
        self.EME = EME()
        self.toGray = torchvision.transforms.Grayscale(1)
        

    def forward(self, y_pred, y):
        predGrey = torch.floor(self.toGray(y_pred * 0.5 + 0.5)*255)
        yGrey = torch.floor(self.toGray(y * 0.5 + 0.5)*255)
        psnr = PSNR(predGrey, yGrey, data_range=255)
        ssim = SSIM(predGrey, yGrey, data_range=255)
        pixdist = self.pixDist(predGrey)
        de = self.DE(predGrey)
        eme = self.EME(predGrey)

        return psnr, ssim, pixdist, de, eme   
    