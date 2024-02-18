# Project Title: Wall Object Removal using Computer Vision

## **Overview:**
This project aims to develop a computer vision model capable of detecting and removing objects present on a wall within an image. By leveraging object detection techniques and inpainting algorithms, the model identifies various items such as television, mirror, clock, etc., on the wall and seamlessly removes them while preserving the overall visual integrity of the image.

## **Key Features:**
- Object Detection: Utilizes state-of-the-art object detection models to identify objects present on the wall.
- Inpainting: Implements advanced inpainting algorithms to intelligently remove detected objects from the image.
- User-friendly Interface: Offers a straightforward interface for users to input images and obtain processed results effortlessly.

## **Dependencies:**
- Python 3.x
- PyTorch
- OpenCV
- NumPy
- PIL (Python Imaging Library)
- Matplotlib
- Transformers
- Diffusers

## **Installation:**
1. Clone the repository:

   ```bash
   git clone https://github.com/username/project.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## **Usage:**
1. Import the necessary modules:

   ```python
   from transformers import pipeline
   import numpy as np
   from PIL import Image
   from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline
   import torch
   import cv2
   from PIL import ImageDraw
   import matplotlib.pyplot as plt
   ```

2. Load the object detection model and image:

   ```python
   checkpoint = "google/owlvit-base-patch32"
   detector = pipeline(model=checkpoint, task="zero-shot-object-detection")
   img, image = get_img("path/to/your/image.jpg")
   ```

3. Get predictions for detected objects:

   ```python
   predictions = get_preds(image)
   ```

4. Mask the detected objects and inpaint the image:

   ```python
   img, masksd = mask_img(image, predictions, img)
   ```

5. Initialize the inpainting pipeline and perform inpainting:

   ```python
   pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting", torch_dtype=torch.float32)
   pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
   ```

6. Save the model:

   ```python
   torch.save(pipeline, "model.h5")
   ```

7. Run the inpainting function to obtain processed images:

   ```python
   inpaintf(pipeline, im_pil, masksd, "output_image.jpg")
   ```

## **Results:**

**Image *before* object detection and removal:**

 ![Frame 1](https://drive.google.com/uc?export=view&id=1JIEHWxmlH9vSxc1lH5VHymqw_Ar-VfxA)

 **Image *after* object detection and removal:**
 
 ![Frame 1](https://drive.google.com/uc?export=view&id=1gi5_C4PKkTxzx1TL0y70VUVbCu0cb6pH)

 ## **Use Cases**

1. **Interior Design Visualization**: Aid in visualizing decor arrangements without physically placing objects.
  
2. **Real Estate Listing Enhancement**: Improve property listing images by removing distracting wall objects.
  
3. **Photography Post-Processing**: Streamline image cleanup for photographers by removing unwanted wall objects.
  
4. **Virtual Staging**: Digitally stage empty rooms with virtual furnishings by removing existing wall objects.
  
5. **Artwork and Design Evaluation**: Evaluate artwork and designs in different environments by removing existing wall objects.
  
6. **Retail Merchandising**: Optimize product placement by digitally removing existing wall displays.
  
7. **Home Renovation Planning**: Experiment with different design ideas without altering physical space by removing existing wall objects.
  
8. **Architectural Visualization**: Present architectural designs in clutter-free environments to clients for better understanding and appreciation.
 
## **Contributing:**
Contributions to this project are welcome! If you have any ideas for improvements or find any issues, please feel free to open an issue or submit a pull request.

## **Acknowledgments:**
- [Transformers](https://huggingface.co/transformers/) for providing pre-trained models.
- [Diffusers](https://github.com/CompVis/diffusion) for the inpainting algorithms.

 
