# soc-vsr-final

This is repository contains the code for SISR (single-image super resolution) using subpixel convolutions.

This can be extended to perform video super resolution by performing super resolution on each video frame as single images.

Files "model.py" and "model1.py" are the 2 model architectures tested. "model.py" gave better performance and is the one
used in the video demonstration.

Training was performed using "train.py" and the parameters obtained post-training were saved in the .pth files.
The 91-images training data is contained in the "training-imgs" folder in the "T91-hr" and "T91-lr" folders.
"nn-model.pth" and "nn-model-1.pth" contain parameters for "model.py".
"nn-model-v2-1.pth", "nn-model-v2-2.pth" and "nn-model-v2-3.pth" contain parameters for "model1.py".

File "test.py" was used to initially test the model on the test images data set and "upscale.py" was used in the video demonstration.
The images used for testing are contained in the "testdata" folder, in the "hr-images" and "lr-images" folders.

File "metrics.py" contains the code used to calculate PSNR (peak signal to noise ratio.)
