import torch
import torchvision
import torch.optim
import os
from Model import LDRMNet, intensityTransform, LDRToTRF
#from Model_imgnet import LDRMNet
import numpy as np
from PIL import Image
import glob
from matplotlib import pyplot as plt

GPU_NUM = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Run using " + str(device))

def lowlight(image_path, filename, outpath):
    data_lowlight = Image.open(image_path)
    data_lowlight_resize = data_lowlight.resize((256, 256))
    lowlight_L = data_lowlight_resize.convert('L')
    data_lowlight_resize = (np.asarray(data_lowlight_resize) / 255.0)
    data_lowlight_resize = torch.from_numpy(data_lowlight_resize).float()
    data_lowlight_resize = data_lowlight_resize * 2.0 - 1.0
    data_lowlight_resize = data_lowlight_resize.permute(2, 0, 1)
    data_lowlight_resize = data_lowlight_resize.to(device).unsqueeze(0)
    
    data_lowlight = (np.asarray(data_lowlight) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight * 2.0 - 1.0
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)
    
    Mapping = intensityTransform(intensities=256, channels=1)
    
    enhancement = LDRMNet()    
    enhancement = enhancement.to(device)
    enhancement.eval()
    enhancement.load_state_dict(torch.load('models/img_ldr/240625_ep1000 LDRMnet img lossldr loss1layers torchmean_sum/ldrmnet-bestpsnr.pth', map_location=device)['model'])
    
    tf = enhancement(data_lowlight_resize)[2]
    enhanced_img = Mapping((data_lowlight, tf))
    enhanced_img = enhanced_img * 0.5 + 0.5

    result_path = os.path.join(outpath, filename)
    plot_path = os.path.join(outpath+'/PLOT', filename)

    tf = tf.squeeze(0)
    tf = tf * 0.5 + 0.5
    tf = tf.cpu().detach().numpy()
    tf = tf * 255
    plt.plot(tf)
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    torchvision.utils.save_image(enhanced_img, result_path)

if __name__ == '__main__':

    with torch.no_grad():
        filePath = 'test input folder here'
        outpath = 'output path folder here'
  
        if not os.path.exists(outpath + '/PLOT'):
            os.makedirs(outpath + '/PLOT')
        file_list = os.listdir(filePath)

        for file_name in file_list:
            image = os.path.join(filePath, file_name)
            print(image)
            lowlight(image, file_name, outpath)

