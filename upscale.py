# Use this to test your model. 
# Use appropriate command line arguments and conditions

import argparse
from pickletools import uint8
from model import Net
# from model1 import Net
import torch
import glob
import PIL.Image as pil_image
import numpy as np
import matplotlib.pyplot as plt
from metrics import calc_psnr
import cv2

def convert_ycbcr_to_rgb(img):
    # Converts an image from ycbcr format to rgb
    (n,m,channels) = img.shape
    if type(img) == np.ndarray:
        for i in range(n):
            for j in range(m):
                (y,cb,cr) = tuple(img[i][j])
                r = y + 1.402*(cr-128)
                g = y - 0.344136*(cb-128) - 0.714136*(cr-128)
                b = y + 1.772*(cb-128)
                img[i][j] = np.array((r,g,b),dtype=int)
        return img

    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)
        img = torch.tensor(img)
        for i in range(n):
            for j in range(m):
                (y,cb,cr) = tuple(img[i][j])
                r = int(y + 1.402*(cr-128))
                g = int(y - 0.344136*(cb-128) - 0.714136*(cr-128))
                b = int(y + 1.772*(cb-128))
                img[i][j] = torch.tensor((r,g,b))
        return img
        
    else:
        raise Exception('Unknown Type', type(img))

def convert_rgb_to_ycbcr(img):
    # Converts an rgb image to ycbcr
    (n,m,channels) = img.shape
    if type(img) == np.ndarray:
        for i in range(n):
            for j in range(m):
                (r,g,b) = tuple(img[i][j])
                y = 16 + 65.738*r/256  + 129.057*g/256 + 25.064*b/256
                cb = 128 - 37.945*r/256 - 74.494*g/256 + 112.439*b/256
                cr = 128 + 112.539/256*r - 94.154*g/256 - 18.295*b/256 
                img[i][j] = np.array((y,cb,cr),dtype=int)
        return img 
    elif type(img) == torch.Tensor:
        if len(img.shape) == 4:
            img = img.squeeze(0)

        for i in range(n):
            for j in range(m):
                (r,g,b) = tuple(img[i][j])
                y = int(16 + 65.738*r/256  + 129.057*g/256 + 25.064*b/256)
                cb = int(128 - 37.945*r/256 - 74.494*g/256 + 112.439*b/256)
                cr = int(128 + 112.539/256*r - 94.154*g/256 - 18.295*b/256) 
                img[i][j] = torch.tensor((y,cb,cr))
        return img 
    else:
        raise Exception('Unknown Type', type(img))



loaded_model = Net()
FILE = "nn-model-1.pth"
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()


parser = argparse.ArgumentParser()

parser.add_argument("--images_dir", help="Location of test images")
parser.add_argument("--patch_size", help="Patch size",type=int)
parser.add_argument("--stride", help="Stride for patches", type=int)

args = parser.parse_args()
ctrl = 1

print("Running started")
for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
    print(ctrl)
    inputImg = pil_image.open(image_path).convert('RGB')
    inputTitle = "Input image: " + image_path
    inputImg.show(inputTitle)
    
    # Load the original upscaled image
    targetImgPath = image_path[:8] + "/hr-images/" + image_path[19:]
    targetImg = np.array(pil_image.open(targetImgPath).convert("RGB"))

    n,m,c = np.array(inputImg).shape
    inputImg = inputImg.resize((2*m,2*n), resample= pil_image.BICUBIC)
    inputImg.show("Bicubic upscaling: "+ image_path)

    #Print the PSNR
    print("Peak signal to noise ratio for bicubic interpolation (in dB): ", calc_psnr(inputImg, targetImg))

    inputImg = np.array(inputImg)
    ycbcrImg = convert_rgb_to_ycbcr(inputImg.copy())
    yChannel = ycbcrImg[:,:,0]

    (n,m,c) = inputImg.shape
    num_patches = (n - args.patch_size + 1)*(m - args.patch_size + 1)
    
    lr_patches = []
    hr_patches = []
    for i in range(0,n - args.patch_size + 1, args.stride):  #Patch size = 4, stride = 1
        for j in range(0,m - args.patch_size + 1, args.stride):
            lr_patches.append(yChannel[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = torch.tensor(np.array(lr_patches), dtype= torch.float)
    
    hr_patches = loaded_model(lr_patches)
    (N,_,n,m) = hr_patches.shape
    hr_patches = hr_patches.view(N,n,m)

    hr_patches = hr_patches.detach().numpy()
    hr_patches = hr_patches.astype('uint8')

    #Reconstructing image from high resolution patches

    (n,m,c) = inputImg.shape

    newYChannel = np.zeros((n,m),dtype=np.uint8)

    for patch_num in range(N):
        x = patch_num % (m - args.patch_size + 1) 
        y = int(patch_num / (n - args.patch_size + 1)) 
        for i in range(args.patch_size):
            for j in range(args.patch_size):
                if y+i<n and x+j < m: 
                    newYChannel[y+i][x+j] = hr_patches[patch_num][i][j]
    
    ycbcrImgNew = ycbcrImg.copy()
    ycbcrImgNew[:,:,0] = newYChannel
    ycrcbImg = ycbcrImg.copy()
    temp = ycrcbImg[:,:,1]
    ycrcbImg[:,:,1] = ycrcbImg[:,:,2]
    ycrcbImg[:,:,2] = temp

    bgrImg = cv2.cvtColor(ycrcbImg, cv2.COLOR_YCR_CB2BGR)
    rgbImg = np.zeros((n,m,3))
    rgbImg[:,:,0] = bgrImg[:,:,2]
    rgbImg[:,:,1] = bgrImg[:,:,1]
    rgbImg[:,:,2] = bgrImg[:,:,0]
    
    rgbImg = np.array(rgbImg, dtype=np.uint8)

    outputImg = pil_image.fromarray(rgbImg)
    outputTitle = "Output image: " + image_path
    
    outputImg.show(outputTitle)

    fileLocation = "image" + str(ctrl) + ".png"

    #Print the PSNR
    print("Peak signal to noise ratio (in dB) after ESPCNN: ", calc_psnr(rgbImg, targetImg))

    # outputImg.save(fileLocation)

    ctrl+=1

    #Line added only for demo
    # if ctrl ==2:
    #     break