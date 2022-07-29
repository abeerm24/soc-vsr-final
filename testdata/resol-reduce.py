'''
 This is the code used to obtain the lower resolution images
'''

import cv2
import glob
import numpy as np

imagesPath = "hr-images"

for image_path in sorted(glob.glob('{}/*'.format(imagesPath))):
    img = cv2.imread(image_path)
    noise = 10*np.random.random(size=img.shape)
    img += noise
    img = np.array(img,dtype=np.uint8)
    resized_img = cv2.resize(img,None,fx = 0.5, fy = 0.5, interpolation=cv2.INTER_CUBIC)
    FILE = "lr-images/" + image_path[10:]
    cv2.imwrite(FILE,resized_img)

