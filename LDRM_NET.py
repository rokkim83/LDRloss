import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torchvision
import torch.optim
import os
import collections
import argparse
import time
import dataloader
import numpy as np
from Model import LDRMNet, LDRMNetTF, LDRToTRF, intensityTransform
from Eval import evaluationLDR
import Criterion

GPU_NUM = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

loss_d_history = collections.deque(maxlen=10)
loss_i_history = collections.deque(maxlen=10)
class lamb_coeff(object):
    def __init__(self):
        self.lamb_d = 1e6
        self.lamb_i = 8.0
    def update(self, val):
        self.lamb_d = val[0]
        self.lamb_i = val[1]

class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# for weight init.
def weightsInit(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

TFconvert = LDRToTRF()

# train func.
def train_one_epoch(trainLoader, model, criterion, optimizer, scheduler, config, lambda_coef):
    model.train()

    epochLossTotal = AverageMeter()
    epochLossimg = AverageMeter()
    epochLossldr = AverageMeter()

    for low, gt, h2d, mask, maskvec in trainLoader:
        low = low.cuda()        
        gt = gt.cuda()
        h2d = h2d.cuda()
        mask = mask.cuda()
        maskvec = maskvec.cuda()
  
        enhancedImage, dl, tf = model(low) #for dl
        enhancedImage = enhancedImage.cuda()
        dl = dl.cuda()

        lossLDR = criterion.ldrLoss(dl, h2d, mask)
        lossImg = criterion.imageLoss(enhancedImage, gt)
    
        totalLoss = lambda_coef.lamb_d * lossLDR + lambda_coef.lamb_i * lossImg
            
        epochLossTotal.update(totalLoss.detach().cpu().numpy(), low[0].size(0))
        epochLossimg.update(lossImg.detach().cpu().numpy(), low[0].size(0))
        epochLossldr.update(lossLDR.detach().cpu().numpy(), low[0].size(0))
    
        optimizer.zero_grad()
        totalLoss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
        optimizer.step()
        scheduler.step()
    
    return [epochLossTotal.avg, epochLossimg.avg, epochLossldr.avg]


# eval. func.
def evaluate(testLoader, model, evals, epoch):
    model.eval()    
    testSummary = [0, 0, 0, 0, 0]

    if epoch % 10 == 0:
        createDirectory(f'outputs/{epoch}')
    indx = 0

    for low, gt in testLoader:
        low = low.cuda()
        gt = gt.cuda()

        enhancedImg = model(low)[0]
        
        psnr, ssim, pixdist, de, eme = evals(enhancedImg.cuda(), gt)
        if epoch % 10 == 0:
            for i in range(enhancedImg.size(0)):
                torchvision.utils.save_image(enhancedImg[i, :, :, :]*0.5 + 0.5, f'outputs/{epoch}/{indx:05d}.png')
                indx += 1
    
        testSummary[0] += psnr.cpu().detach().numpy()
        testSummary[1] += ssim.cpu().detach().numpy()
        testSummary[2] += pixdist.cpu().detach().numpy()
        testSummary[3] += de.cpu().detach().numpy()
        testSummary[4] += eme.cpu().detach().numpy()

    testSummary[0] /= len(testLoader)
    testSummary[1] /= len(testLoader)
    testSummary[2] /= len(testLoader)
    testSummary[3] /= len(testLoader)
    testSummary[4] /= len(testLoader)

    return testSummary

def train(config):
    sum_time = 0

    if torch.cuda.is_available():
        cudnn.benchmark = True
    else:
        raise Exception("No GPU found, please run without --cuda")

    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)  # change allocation of current GPU
    enhancement = LDRMNet().cuda()
    enhancement.apply(weightsInit)
    enhancement = enhancement.cuda()

    trainDataset = dataloader.trainLoader(config.train_images_path)
    trainLoader = torch.utils.data.DataLoader(dataset=trainDataset, batch_size=config.train_batch_size, shuffle=True,
                                               num_workers=config.num_workers, pin_memory=True)

    testDataset = dataloader.testLoader(config.test_images_path)
    testLoader = torch.utils.data.DataLoader(dataset=testDataset, batch_size=config.train_batch_size, shuffle=False, 
                                              num_workers=config.num_workers, pin_memory=True)

    criterion = Criterion.criterion().cuda()                                          
            
    optimizer = torch.optim.AdamW(enhancement.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(trainLoader) * config.num_epochs, 0)

    #enhancement.train()
    evals = evaluationLDR().cuda()
    lambda_coef = lamb_coeff()
    
    num_params = 0
    for param in enhancement.parameters():
        num_params += param.numel()
    print("==================================================")
    print("||\ttranining mode: image network training   ||")
    print('||\t# of params : %d\t\t\t\t   ||' % num_params)
    print("==================================================")

    bestPPsnr = 0
    bestPSsim = 0
    bestPPixdist = 0
    bestPEME = 0
    bestPDE = 0

    bestSPsnr = 0
    bestSSsim = 0
    bestSPixdist = 0
    bestSEME = 0
    bestSDE = 0

    balancedPSNR = 0
    balancedSSIM = 0
    balancedPixDist = 0
    balancedEME = 0
    balancedDE = 0

    for epoch in range(config.num_epochs):
        st = time.time()
        print()
        print("=========== Epoch " + str(epoch+1) + " ===========")

        totalLoss = train_one_epoch(trainLoader, enhancement, criterion, optimizer, scheduler, config, lambda_coef)
        testSummary = evaluate(testLoader, enhancement, evals, epoch)

        if len(loss_i_history) >= 4 :
            mean_loss_i = np.mean(list(loss_i_history)[-4:])
            mean_loss_d = np.mean(list(loss_d_history)[-4:])
            rate_i = loss_i_history[-1] / mean_loss_i
            rate_d = loss_d_history[-1] / mean_loss_d

            lamb_i = rate_i / (rate_i + rate_d)
            lamb_d = rate_d / (rate_i + rate_d)
            lambda_coef.update([lamb_d, lamb_i])


        save_dict ={
            "epoch" : epoch + 1,
            "psnr" : testSummary[0],
            "ssim" : testSummary[1],
            "pixdist" : testSummary[2],
            "de" : testSummary[3],
            "eme" : testSummary[4],
            "bestpsnr": bestPPsnr,
            "bestssim": bestPSsim,
            "bestpixdist": bestPPixdist,
            "bestDE" : bestPDE,
            "bestEME": bestPEME, 
            "model" : enhancement.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if testSummary[0] > 20 and testSummary[2]*testSummary[3]*testSummary[4] > balancedPixDist *balancedEME * balancedDE and testSummary[0]*testSummary[1] > balancedPSNR *balancedSSIM:
            balancedPSNR = testSummary[0]
            balancedSSIM = testSummary[1]
            balancedPixDist = testSummary[2]
            balancedDE = testSummary[3]
            balancedEME = testSummary[4]            

            save_dict["bestpsnr"] = balancedPSNR
            save_dict["bestssim"] = balancedSSIM
            save_dict["bestpixdist"] = balancedPixDist
            save_dict["bestDE"] = balancedDE
            save_dict["bestEME"] = balancedEME            

            print(f"****** best performances: PSNR={balancedPSNR}, SSIM={balancedSSIM}, PixDist={balancedPixDist}, DE={balancedDE}, EME={balancedEME}")

            checkpointPath = f"{config.checkpoint_dir}/ldrmnet-Balanced.pth"
            torch.save(save_dict, checkpointPath)

        if testSummary[2]*testSummary[3]*testSummary[4] > bestPPixdist *bestPDE * bestPEME:
            bestPPsnr = testSummary[0]
            bestPSsim = testSummary[1]
            bestPPixdist = testSummary[2]
            bestPDE = testSummary[3]
            bestPEME = testSummary[4]            

            save_dict["bestpsnr"] = bestPPsnr
            save_dict["bestssim"] = bestPSsim
            save_dict["bestpixdist"] = bestPPixdist
            save_dict["bestDE"] = bestPDE
            save_dict["bestEME"] = bestPEME            

            print(f"****** best performances: PSNR={bestPPsnr}, SSIM={bestPSsim}, PixDist={bestPPixdist}, DE={bestPDE}, EME={bestPEME}")

            checkpointPath = f"{config.checkpoint_dir}/ldrmnet-bestcontrast.pth"
            torch.save(save_dict, checkpointPath)

        if testSummary[0]*testSummary[1] > bestSPsnr *bestSSsim:
            bestSPsnr = testSummary[0]
            bestSSsim = testSummary[1]
            bestSPixdist = testSummary[2]
            bestSDE = testSummary[3]
            bestSEME = testSummary[4]            

            save_dict["bestpsnr"] = bestSPsnr
            save_dict["bestssim"] = bestSSsim
            save_dict["bestpixdist"] = bestSPixdist
            save_dict["bestDE"] = bestSDE
            save_dict["bestEME"] = bestSEME            

            print(f"****** best PSNR and SSIM: PSNR={bestSPsnr}, SSIM={bestSSsim}, PixDist={bestSPixdist}, DE={bestSDE}, EME={bestSEME}")

            checkpointPath = f"{config.checkpoint_dir}/ldrmnet-bestpsnr.pth"
            torch.save(save_dict, checkpointPath)
        
        checkpointPath = f"{config.checkpoint_dir}/ldrmnet-latest.pth"
        torch.save(save_dict, checkpointPath)

        et = time.time() - st
        print('%d/%d epoch: %.3f sec' % (epoch + 1, config.num_epochs, et))
        print(f'Loss, ImageLoss, LDRLoss:{totalLoss}')
        print(f"current performances(Per): PSNR={testSummary[0]}/{bestPPsnr}, SSIM={testSummary[1]}/{bestPSsim}, PixDist={testSummary[2]}/{bestPPixdist}, DE={testSummary[3]}/{bestPDE}, EME={testSummary[4]}/{bestPEME}")
        print(f"current performances(Sim): PSNR={testSummary[0]}/{bestSPsnr}, SSIM={testSummary[1]}/{bestSSsim}, PixDist={testSummary[2]}/{bestSPixdist}, DE={testSummary[3]}/{bestSDE}, EME={testSummary[4]}/{bestSEME}")
        sum_time += et
        rTime = (sum_time / (epoch + 1)) * (config.num_epochs - (epoch + 1))
        print("Estimated time remaining :%d hour %d min %d sec" % (
            rTime / 3600, (rTime % 3600) / 60, (rTime % 3600) % 60))

def LDRMNetTrain(train_image_path="./data/train_data/",
                   test_image_path="./data/test_data/",       
                   learning_rate=0.001,
                   weight_decay=0.01,
                   grad_clip_norm=0.1,
                   beta=1,
                   epochs=1000,
                   batch_size=32,
                   num_workers=5,
                   checkpoint_dir="models/"):

    start_time = time.time()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--train_images_path', type=str, default=train_image_path)
    parser.add_argument('--test_images_path', type=str, default=test_image_path)
    parser.add_argument('--lr', type=float, default=learning_rate)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--grad_clip_norm', type=float, default=grad_clip_norm)
    parser.add_argument('--beta', type=int, default=beta)
    parser.add_argument('--num_epochs', type=int, default=epochs)
    parser.add_argument('--train_batch_size', type=int, default=batch_size)
    parser.add_argument('--num_workers', type=int, default=num_workers)
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir)

    config = parser.parse_args()

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    train(config)
    total_time = time.time() - start_time
    print("total: %dhour %dmin %dsec" % (total_time / 3600, (total_time % 3600) / 60, (total_time % 3600) % 60))


if __name__ == "__main__":
    LDRMNetTrain(train_image_path=f"train folder here",
                 test_image_path=f"test folder here",
                 epochs=500, batch_size=32)