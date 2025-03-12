import sys
import os
import io
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class ImageSearchGUI(QWidget):

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Search Inputs")
        self.setStyleSheet("background-color: #2c3e50;")
        layout = QVBoxLayout()

        title_label = QLabel("Image Search UI")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setStyleSheet("color: #ecf0f1;")
        layout.addWidget(title_label)

        test_image_layout = QHBoxLayout()
        test_image_label = QLabel("Test Image Path:")
        test_image_label.setStyleSheet("color: #ecf0f1; font-size: 14px;")
        test_image_layout.addWidget(test_image_label)

        self.image_path_input = QLineEdit()
        self.image_path_input.setPlaceholderText("Enter test image path")
        self.image_path_input.setStyleSheet(self.input_style())
        test_image_layout.addWidget(self.image_path_input)

        browse_image_button = QPushButton("Browse")
        browse_image_button.setFont(QFont("Arial", 12))
        browse_image_button.setStyleSheet(self.button_style())
        browse_image_button.clicked.connect(self.browse_image)
        test_image_layout.addWidget(browse_image_button)
        layout.addLayout(test_image_layout)

        folder_path_layout = QHBoxLayout()
        folder_path_label = QLabel("Folder Path:")
        folder_path_label.setStyleSheet("color: #ecf0f1; font-size: 14px;")
        folder_path_layout.addWidget(folder_path_label)

        self.folder_path_input = QLineEdit()
        self.folder_path_input.setPlaceholderText("Enter folder path")
        self.folder_path_input.setStyleSheet(self.input_style())
        folder_path_layout.addWidget(self.folder_path_input)

        browse_folder_button = QPushButton("Browse")
        browse_folder_button.setFont(QFont("Arial", 12))
        browse_folder_button.setStyleSheet(self.button_style())
        browse_folder_button.clicked.connect(self.browse_folder)
        folder_path_layout.addWidget(browse_folder_button)
        layout.addLayout(folder_path_layout)

        self.search_button = QPushButton("Submit")
        self.search_button.setFont(QFont("Arial", 14))
        self.search_button.setStyleSheet("background-color: #3498db; color: white; border-radius: 5px;")
        self.search_button.clicked.connect(self.search_action)
        layout.addWidget(self.search_button, alignment=Qt.AlignCenter)

        # Info Tab
        self.info_tab = QTextEdit()
        self.info_tab.setReadOnly(True)
        self.info_tab.setStyleSheet("background-color: #1e272e; color: #ecf0f1; padding: 5px; border-radius: 5px; font-size: 12px;")
        layout.addWidget(self.info_tab)

        self.setLayout(layout)
        self.setGeometry(200, 200, 400, 400)

    def input_style(self):
        return "color: #34495e; background-color: #ecf0f1; padding: 5px; border-radius: 5px; font-size: 14px;"

    def button_style(self):
        return "background-color: #1abc9c; color: white; padding: 5px; border-radius: 5px; font-size: 12px; font-weight: bold;"

    def browse_image(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if image_path:
            self.image_path_input.setText(image_path)
            self.update_info(f"Selected Image: {image_path}")

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path_input.setText(folder_path)
            self.update_info(f"Selected Folder: {folder_path}")

    def search_action(self):
        image_path = self.image_path_input.text()
        folder_path = self.folder_path_input.text()
        self.worker = ImageSearchWorker(image_path, folder_path)


        if not image_path or not folder_path:
            self.update_info("Please enter both image and folder paths.")
            return

        self.search_button.setEnabled(False)

        self.update_info("Starting search...")

        try:
            self.worker = ImageSearchWorker(image_path, folder_path)
            self.worker.finished.connect(self.on_search_finished)
            self.worker.start()
        except Exception as e:
            self.update_info(f"Error: {e}")
            self.search_button.setEnabled(True)

    def on_search_finished(self, message):
        self.update_info(message)
        self.search_button.setEnabled(True)

    def update_info(self, message):
        print(message)  # Output to IDE
        self.info_tab.append(message)  # Output to GUI



class ImageSearchWorker(QThread):
    finished = pyqtSignal(str)

    def __init__(self, test_path, folder_path ):
        super().__init__()
        self.test_path = test_path
        self.folder_path = folder_path
        

    def run(self):
        try:
            objMain = imgSearch(self.test_path, self.folder_path, similarity=0.8)
            objMain.main()
            self.finished.emit("Search Completed!")
        except Exception as e:
            self.finished.emit(f"Error: {str(e)}")



class imgSearch:

    def __init__(self, testPath, folderPath, similarity):
        self.testPath = testPath
        self.folderPath = folderPath
        self.similarity = similarity
        self.model = None
        self.processor = None
        
    
    def loadModal(self):
        try:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="C:\\Users\\Admin\\clip_model")
            # processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir="C:\\Users\\Admin\\clip_model")
            #model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor =CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="C:\\Users\\Admin\\clip_model") #CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            exit(1)

    def imgLoad(self):
        try:
            test_img = Image.open(self.testPath).convert("RGB")
            print("Test image loaded successfully.")
        except Exception as e:
            print(f"Error loading test image: {e}")
            return None, None

        image_list = self.imgCount(self.folderPath)
        return image_list, test_img

    def imgCount(self, folder_Path):
        image_list = []
        try:
            for filename in os.listdir(folder_Path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_Path, filename)
                    image = Image.open(img_path).convert("RGB")
                    image_list.append(image)
            print(f"Loaded {len(image_list)} images.                       #Counted(*needed to be removed in future)")
        except Exception as e:
            print(f"Error loading images from folder: {e}")
            return []
        return image_list

    def comparisonProcess(self, image_list, test_img):
        if not image_list:
            print("No images found for comparison.")
            return

        if torch.cuda.is_available():
            device = "cuda"
        else :
            device = "cpu"
            
        self.model.to(device)

        test_Input = self.processor(images=test_img, return_tensors="pt").to(device)
        inputs = self.processor(images=image_list, return_tensors="pt").to(device)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            test_img_features = self.model.get_image_features(**test_Input)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        test_img_features = test_img_features / test_img_features.norm(dim=-1, keepdim=True)

        print("The number of images you have uploaded is", len(image_list))

        for i in range(len(image_list)):
            similarity = torch.nn.functional.cosine_similarity(
                test_img_features, image_features[i:i+1], dim=-1
            )

            if similarity.item() > 0.80:
                print(f"Similarity between test image and Image {i + 1} in the folder u uploded: {similarity.item():.4f}")
                image_list[i].show()  # Directly display the image


    def main(self):
        self.loadModal()
        image_list, test_img = self.imgLoad()
        if test_img is None or not image_list:
            return 0. 
        self.comparisonProcess(image_list, test_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageSearchGUI()
    window.show()
    sys.exit(app.exec_())
