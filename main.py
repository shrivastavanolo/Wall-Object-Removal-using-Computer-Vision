import torch
import helper as df

#preprocess image using cv2 and PIL
img,image=df.get_img(r"<path-to-image>")

#Get predictions for candidate labels in image
predictions=df.get_preds(image)

#Get masks for detected objects
img,masksd=df.mask_img(image,predictions,img)
im_pil = df.convert_cv_img(img)

#load model
pipeline=torch.load(r"path-to-model/model.h5")

#Inpaint the image and save as <image name>
df.inpaintf(pipeline,im_pil,masked,"<image-name>")
