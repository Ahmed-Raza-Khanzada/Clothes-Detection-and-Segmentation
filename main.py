from detection import detect
from segmentation import segmentation
import os
import shutil

import requests

def make_segmentation(url,model_path= "detection_v8_models/100epochsbest.pt",my_cloths = ["shirt","nodetect","jacket","dress"]):
    
    try:
        try:
            img_url = url
            path = "temp_save"
            if not os.path.exists(path):
                os.mkdir(path)
            image_name = "mycloth.jpg"
            response = requests.get(url)
            with open(f"{path}/{image_name}", "wb") as f:
                f.write(response.content)
            del response
            print("Image downloaded successfully")
        except Exception as e:
            print(e)
            print("Error in downloading image")
            print("Please try again with different image url")
            print("Below is incorrect image url")
            print(img_url)
            return
        
        r = detect(f"{path}/{image_name}",model_path=model_path)
        # r = {"nodetect":[f"temp_save/{image_name}"]}
        images_out_paths = []
        for cloth_name,images_paths in r.items():
            # print(cloth_name,images_paths)
            if cloth_name in my_cloths:
                for i in range(len(images_paths)):
                    image = images_paths[i]
                    if cloth_name in my_cloths:
                        # continue
                        image_ouput = segmentation(image,f"{image_name[:-4]}_100epochs_{i}")
                    else:
                        image_ouput = segmentation(image,f"{image_name[:-4]}__withoutcropping_100epochs_{i}")
                    images_out_paths.append(image_ouput)
        if os.path.exists("runs/detect/predict"):
            shutil.rmtree("runs/detect/predict")
        return images_out_paths
    except Exception as e:
        print(e)
        if os.path.exists(path):
            shutil.rmtree(path)
        if os.path.exists("runs/detect/predict"):
            shutil.rmtree("runs/detect/predict")
        os.mkdir(path)


if __name__ == "__main__":
    url = "https://m.media-amazon.com/images/I/51FbtoBDKyL._AC_UY1100_.jpg"
    print(make_segmentation(url,model_path= "detection_v8_models/100epochsbest.pt"))