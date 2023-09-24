import numpy as np
import cv2
import os
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model
def segmentation(image,output_name = ""):
    model = create_model("Unet_2020-10-30")
    model.eval()
    # path = img_path
    # image = load_rgb(path.split("/")[-1])
    image= load_rgb(image)
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    with torch.no_grad():
        prediction = model(x)[0][0]
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    mask_image = np.zeros_like(image)
    print(mask_image.shape,image.shape)
    mask_image[:, :, 0] = mask * image[:, :, 0]
    mask_image[:, :, 1] = mask * image[:, :, 1]
    mask_image[:, :, 2] = mask * image[:, :, 2]
    # dst = cv2.addWeighted(image, 1, (cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * (0, 255, 0)).astype(np.uint8), 0.5, 0)
    # stack_image = np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * 255, dst,mask_image])
    inc1 = 0
    list1 = os.listdir("static")
    while True:
      

        if f"{output_name}.jpg" in list1:
            output_name = output_name[:-1]+str(inc1)
            if f"{output_name}.jpg" in list1:
                inc1+=1
                continue
            else:
                
                break
        else:
            break
    
    print("*"*40)
    print("Segmentation Results saved at: ",f'static/{output_name}.jpg')
    print("*"*40)
    
    cv2.imwrite(f'static/{output_name}.jpg',cv2.cvtColor(mask_image,cv2.COLOR_BGR2RGB))
    return f'static/{output_name}.jpg'