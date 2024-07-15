from tracemalloc import start
import torch
import torch.nn.functional as F
import torch.nn as nn

class h2dLossSep(nn.Module):
    def __init__(self, **kwargs):
        super(h2dLossSep, self ).__init__(**kwargs)
        
        self.matA = torch.zeros(32640, 255)
        startIndex = 0
        for layer in range(1, 256): #layer 1 to 255 (255)
            self.matA[startIndex:startIndex+(256-layer), ...] = torch.ones(256-layer, 255) - torch.triu(torch.ones(256-layer, 255), layer) -torch.tril(torch.ones(256-layer, 255), -1)
            startIndex = startIndex+(256-layer)
        self.matA = self.matA.cuda()

    def forward(self, d1layer, hvec, mask): # mask = mask_l in dataloader

        batchSize = d1layer.size()[0]

        mask = torch.pow(mask, 2.5) # powered by alpha

        startIndex = 0
        loss1Layers = torch.zeros(255, ).cuda()
        matAext = self.matA.expand(batchSize, 32640 , 255) # extend matA according to batchsize
        
        for layer in range(1, 256):
            Al = matAext[:, startIndex:startIndex+(256-layer), ...]
            matAd = torch.matmul(Al, torch.swapaxes(d1layer, 1, 2)).squeeze(2)
            vecH = hvec[:, startIndex:startIndex+(256-layer)]

            loss1Layers[layer-1] = torch.mean(torch.pow(matAd - vecH, 2)* mask[:, layer-1].unsqueeze(1).expand(-1, 256-layer))#240618 org #The expanded size of the tensor (255) must match the existing size (32) at non-singleton dimension 0.  Target sizes: [255].  Tensor sizes: [32]
            #if torch.sum(torch.isnan(loss1Layers[layer-1])) > 0:
            #    check = 1
            startIndex = startIndex + (256-layer)
        
        return torch.mean(loss1Layers)
       
class criterion(nn.Module):
    def __init__(self, **kwargs):
        super(criterion, self).__init__(**kwargs)
        self.imageLoss = torch.nn.MSELoss().cuda()
        self.ldrLoss = h2dLossSep().cuda()
