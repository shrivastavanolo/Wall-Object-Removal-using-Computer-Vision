import torch
import helper as df

img,image=df.get_img(r"<path-to-image>")
predictions=df.get_preds(image)
img,masksd=df.mask_img(image,predictions,img)
im_pil = df.convert_cv_img(img)

pipeline=torch.load(r"path-to-model/model.h5")
df.inpaintf(pipeline,im_pil,masked,"<image-name>")
