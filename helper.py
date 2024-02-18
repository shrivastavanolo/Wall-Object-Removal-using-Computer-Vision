from transformers import pipeline
checkpoint = "google/owlvit-base-patch32"
detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
import numpy as np
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from diffusers import StableDiffusionInpaintPipeline
import torch
import cv2
from PIL import ImageDraw
import matplotlib.pyplot as plt

def get_img(path):
    cv_img=cv2.imread(path)
    cv_img=cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) ############################ cv img

    pil_img = Image.open(path)
    pil_img = Image.fromarray(np.uint8(pil_img)).convert("RGB")    ###pil img
    return cv_img,pil_img

# img,image=get_img(r"C:\Users\shrey\OneDrive\Documents\sigmoid\image-processing\wall images\8.jpg")

def get_preds(image):
#prediction detector
    predictions = detector(
        image,
        candidate_labels=["television","tv","box","mirror","decor","clock","frame","cable","air conditioner","white rectangle","black rectangle"]
    )
    return predictions

# predictions=get_preds(image)
# mask2 for wall objects (cv2) 
# masksd for floor objects (stable diffusion)

def mask_img(image,predictions,img):
    masksd=image.copy()
    draw1 = ImageDraw.Draw(masksd)
    h,w=image.size
    draw1.rectangle((0,0,h,w),fill="black")
    iarea=h*w
    for prediction in predictions:
        if prediction["score"]>=0:
            box = prediction["box"]

            xmin, ymin, xmax, ymax = box.values()
            area= (ymax-ymin)*(xmax-xmin)
            if area<=iarea/5:
                open_cv_mask = np.zeros(img.shape[:2], dtype="uint8")
                cv2.rectangle(open_cv_mask, (xmin-10, ymin-10), (xmax+10, ymax+10), 255, -1)
                img = cv2.inpaint(img, open_cv_mask, inpaintRadius=7, flags=cv2.INPAINT_NS)
            else:
                draw1.rectangle((xmin, ymin, xmax, ymax), outline="white", width=5,fill="white")
                
    return img, masksd

# img,masksd=mask_img(image,predictions,img)

def convert_cv_img(img):
    return Image.fromarray(img)

# im_pil = convert_cv_img(img)

# pipeline = StableDiffusionInpaintPipeline.from_pretrained(
#     "runwayml/stable-diffusion-inpainting",
#     torch_dtype=torch.float32,
# )

# pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
# torch.save(pipeline,"model.h5")
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is

def inpaintf(pipeline,im_pil,masksd,name,prompt="background:0.5 wall and floor"):
    image1 = pipeline(prompt=prompt,image=im_pil,mask_image=masksd,num_inference_steps=15).images[0]
    plt.imshow(image1)
    plt.show()

    image1.save(name)

# inpaintf(pipeline,im_pil,masksd,"hi8.jpeg")
