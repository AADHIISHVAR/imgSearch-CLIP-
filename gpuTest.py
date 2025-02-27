import torch
import torchvision
import transformers 
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import IPython

# Check versions of important libraries
print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Transformers version:", transformers.__version__)
print("Pillow version:", Image.__version__)
print("IPython version:", IPython.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("___________________________________________________________________________________________________________________")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create a tensor and move it to the GPU
print('Sample programn to check cuda is workin perfect')
x = torch.rand(3, 3).to(device)
print(x)


#Attempt to load the local CLIP model
try:
    print("loding modal")
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="C:/Users/abhej/.cache/huggingface/models/models--openai--clip-vit-base-patch32")
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="C:/Users/abhej/.cache/huggingface/models/models--openai--clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="C:/Users/abhej/.cache/huggingface/models/models--openai--clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="C:/Users/abhej/.cache/huggingface/models/models--openai--clip-vit-base-patch32")

    print("CLIP model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
