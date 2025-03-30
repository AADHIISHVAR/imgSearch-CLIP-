its wrong purposefully , ill update it later
# # imgSearch-CLIP


**imgSearch-CLIP** is a local image search tool powered by OpenAI's CLIP model. This software allows users to search for images on their local machine using a reference image. By leveraging the power of CLIP, imgSearch-CLIP can find visually or semantically similar images with high accuracy.


## Features

- **Local Image Search**: Finds images from a dataset stored locally.
- **Reference-Based Search**: Users can input an image to find similar images.
- **Fast and Efficient**: Utilizes CLIP embeddings for quick similarity comparison.
- **No Internet Required**: Runs entirely offline, ensuring privacy and security.


## How It Works

1. The reference image is processed through the CLIP model to extract feature embeddings.
2. The stored images in the dataset are also pre-processed to obtain their embeddings.
3. The embeddings are compared using cosine similarity to find the most relevant matches.
4. The best-matching images are displayed as search results.

![diagram](https://github.com/user-attachments/assets/0fdb99ba-4e83-48eb-b197-baed32815d08)


## Installation

   RUN THE gpuTest.py @ cmd administrator 
               (or)
   read the gpuTest.py 's description

### Prerequisites

Ensure you have the following installed:

- Python (>=3.8)
- PyTorch
- OpenAI CLIP
- NumPy
- PIL (Pillow)
- CUDA


### Steps to Install

```bash
# Clone the repository
git clone [https://github.com/yourusername/imgSearch-CLIP.git](https://github.com/yourusername/imgSearch-CLIP.git)
cd imgSearch-CLIP


# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


# Install dependencies
pip install -r requirements.txt  #i'll update it later
```


## Usage

1. Place the images you want to search within the `dataset/` directory.
2. Run the script and provide the path to your reference image:

```bash
python search.py --image path/to/reference.jpg
```

3. The most similar images will be displayed as results.


## Example

```bash
python search.py --image sample.jpg
```

*Output:* List of matching images ranked by similarity.


## Future Enhancements

- GUI support for easy image selection
- Support for text-based image search
- Optimization for large-scale datasets
- A Dockerized version of this software will be uploaded soon by me.


## Contributing

Feel free to fork the repository and submit pull requests for improvements!





---

**Developed with ❤️ using OpenAI's CLIP**


## What is CLIP?

 It enables the model to understand images and text together, making it useful for tasks like image classification, retrieval, and zero-shot learning without needing specialized datasets.


It uses a transformer architecture to analyze the image, extract its features, and convert them into input for the neural network model.

In some cases:

When the image search is performed on a larger dataset, it takes more time to find the image while using a CPU. To improve performance, I have utilized CUDA to run the neural network on NVIDIA's GPU.


For example:

how ur hardware affects my softwares performance

To find an image in a dataset of 700 images, running on a CPU (Intel i5 3rd Gen U-series) takes over 400 seconds to complete the search.

However, when the search is performed using a GPU (NVIDIA GeForce 3050 6GB), it takes just 77 seconds to complete the search on an image dataset consisting of over 1000 images. Additionally, it is capable of opening the searched images sequentially.

