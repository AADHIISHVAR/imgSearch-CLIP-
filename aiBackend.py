# import app
from idlelib.query import Query

from fastapi import FastAPI,Query
from fastapi.middleware.cors import CORSMiddleware


import torch
import torchvision
import transformers
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import IPython
import os
import sys

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Include OPTIONS method
    allow_headers=["*"],
)


model_path = os.environ.get('MODEL_PATH')

@app.get("/")
def read_root():
    return {"message": "AI Model Backend Running"}

@app.get("/run_test")
def run_test():
    print("Running tests")
    versionData = {
        "Torch version": torch.__version__,
        "Torchvision version": torchvision.__version__,
        "Transformers version": transformers.__version__,
        "Pillow version": Image.__version__,
        "IPython version": IPython.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "AI model loadel test loding done": "no prob"
    }

    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=model_path)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32",
                                                  cache_dir=model_path,use_fast=True)
        print("CLIP model and processor loaded successfully.")
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        return {"error": f"Error loading CLIP model: {str(e)}"}  # Return an error response

    return versionData  # Return only if no error occurs

#imgCount=0
@app.post("/startSearch")
def startSearch(
    testPath: str = Query(...),
    folderPath: str = Query(...),
    sensitivity: float = Query(...)
):
    global info, model
    info={}

    try:

        print("Loading model...")
        info.update({"Loading model...":"Green flag"})

        # Ensure cache directory exists

        # cache_path = model_path
        # os.makedirs(cache_path, exist_ok=True)

        # Load model and processor
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir=model_path)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=model_path,use_fast=True)

        print("Model loaded successfully.")
        info.update({"Model loaded successfully.":"Green flag"})

    except Exception as e:
        error_message = f"Error loading model: {e}"
        print(error_message)
        info.update({f"Error loading model: {e}":"Red flag"})

    try:
        test_img = Image.open(testPath).convert("RGB")
        print("Test image loaded successfully.")
        info.update({"Test image loaded successfully.":"Green flag"})
    except Exception as e:
        print(f"Error loading test image: {e}")
        info.update({f"Error loading test image: {e}":"Red flag"})

    image_list = []
    try:
        for filename in os.listdir(folderPath):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folderPath, filename)
                image = Image.open(img_path).convert("RGB")
                image_list.append(image)

        #return {f"Loaded {len(image_list)} images.": "green flag"}

    except Exception as e:
        print(f"Error loading images from folder: {e}")
        info.update({f"Error loading images from folder: {e}":"Red flag"})

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model.to(device)

    test_Input = processor(images=test_img, return_tensors="pt").to(device)
    inputs = processor(images=image_list, return_tensors="pt").to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        test_img_features = model.get_image_features(**test_Input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    test_img_features = test_img_features / test_img_features.norm(dim=-1, keepdim=True)

    print("The number of images you have uploaded is", len(image_list))
    #imgCount = len(image_list)
    okVariable = None  # Initialize before the loop

    for i in range(len(image_list)):
        similarity = torch.nn.functional.cosine_similarity(
            test_img_features, image_features[i:i + 1], dim=-1
        )

        if similarity.item() >= sensitivity:
            print(f"Similarity between test image and Image {i + 1} in the uploaded folder: {similarity.item():.4f}")
            info.update({f"Similarity between test image and Image {i + 1} in the uploaded folder": round(
                similarity.item(), 4)})
            print("# The similarity is rounded up to the nearest even number because it's Python (although that's not literally true)")
            image_list[i].show()  # Display the image
            print(sensitivity)
            okVariable = "ok"
# if the image is not found
    if okVariable != "ok":
        info.update({f"No similar image found to the simaliraty score tyr changing teh sensitivity value from teh current value{sensitivity}":"Red flag"})
        # print(f"No similar image found to the simaliraty score tyr changing teh sensitivity value from teh current value{sensitivity}")
        info.update({"None of the images were similar to the test image": "Red flag"})
        # 💡 You can handle further actions here
        # e.g., log, alert, or provide fallback visuals/messages


@app.post("/imageCount")
def imageCount(testPath: str, folderPath: str):
    imgCountList=0
    for filename in os.listdir(folderPath):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            #img_path = os.path.join(folderPath, filename)
            #image = Image.open(img_path).convert("RGB")
            imgCountList = imgCountList+1
    print(f"The no of images in the folder u uploded is {imgCountList}")
    return {"totalImages":imgCountList}

@app.get("/backendMessage")
def backendMessage():
    print(info)
    return info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
