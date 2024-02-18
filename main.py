import torch
import helper as df

img,image=df.get_img(r"C:\Users\shrey\OneDrive\Documents\sigmoid\wall images\8.jpg")
predictions=df.get_preds(image)
img,masksd=df.mask_img(image,predictions,img)
im_pil = df.convert_cv_img(img)

pipeline=torch.load(r"C:\Users\shrey\OneDrive\Documents\sigmoid\model.h5")
df.inpaintf(pipeline,im_pil,masksd)