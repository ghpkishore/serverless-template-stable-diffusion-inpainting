import banana_dev as banana
import base64
from io import BytesIO
from PIL import Image
import requests
import random

def loadAndDownloadImages(image_path,mask_path):
    image_load = Image.open(requests.get(image_path, stream=True).raw)
    image=image_load.convert('RGB').copy()
    masked_image_load=Image.open(requests.get(mask_path, stream=True).raw)
    masked_image=masked_image_load.convert('RGB').copy()
    return image,masked_image

def imgToBase64String(filename):
    img = Image.open(filename)
    im_file = BytesIO()
    img.save(im_file, format="PNG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes).decode('utf-8')
    return im_b64



init_image_base64 = imgToBase64String("yogabar2.png")
masked_image_base64 = imgToBase64String("yogabar2Mask.png")  
seed=random.randint(0,109471801)
prompt='Fruits and Nuts'
# model_inputs = {"prompt":"","init_image_base64":init_image_string,"mask_image_base64":mask_image_string,"strength":0.6,"guidance_scale":75,"steps":50}
model_inputs = {
    'prompt':prompt,
    'init_image_base64':init_image_base64,
    'mask_image_base64':masked_image_base64,
    'guidance_scale':7.5,
    'steps':20,
    'height':512,
    'width':512,
}

#Call model deployed on banana
api_key = "3c003ff4-ac8b-4a10-ac33-afdfe331ca26"
model_key = "aa9fe10e-cf53-4217-9445-17a32033370c"
output = banana.run(api_key,model_key,model_inputs)
print(output)
try:
    output_image_string = output["modelOutputs"][0]["image_base64"]
    print("Post-processed image")
    image_encoded = output_image_string.encode('utf-8')
    print("Image Encoded")
    image_bytes = BytesIO(base64.b64decode(image_encoded))
    print(" Image in Bytes")
    image = Image.open(image_bytes)
    print("Image opened")
    image.save("output.png")
    print("Image saved")
except:
    print("Error")
#Call the model locally
# import requests
# res = requests.post('http://localhost:8000/', json = model_inputs)

# print(res.json())