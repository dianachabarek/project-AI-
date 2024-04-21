import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random

# function for loading images in opencv
# resolution too high -> downsizing the images 

def load_image(path, target_size=(256, 512)):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.resize(img, target_size)
    return img

# overlays the pedicted segmentatuion on the original image 
def overlay_mask_on_image(image, mask, threshold=0.5, color=[0, 255, 0], alpha=0.4):
   
    binary_mask = mask > threshold
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[..., i] = np.where(binary_mask, color[i], 0)

    overlayed_image = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlayed_image

# create predictions and visualize them
def predict_and_visualize(model_path, base_path):
    model = load_model(model_path)
    
    city_folders = [os.path.join(base_path, folder) for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
    selected_images = []
    # choose 5 random images from the test folder 
    for city_folder in random.sample(city_folders, 5):  
        image_files = [os.path.join(city_folder, f) for f in os.listdir(city_folder) if f.endswith('.png')]
        if image_files:
            selected_image = random.choice(image_files)  
            selected_images.append(selected_image)

    plt.figure(figsize=(15, 10))

    #test if we are able to load the random images, if not error  
    for i, image_path in enumerate(selected_images):
        image = load_image(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            continue
        
        # apply the predictions onto the 5 random images 
        image_expanded = np.expand_dims(image, axis=0)
        prediction = model.predict(image_expanded)
        predicted_mask = prediction.squeeze()

        
        overlayed_image = overlay_mask_on_image(image, predicted_mask, threshold=0.5, color=[0, 255, 0], alpha=0.4)

        # plot the random images 
        plt.subplot(3, 5, i + 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # plot the predicted images 
        plt.subplot(3, 5, i + 6)
        plt.imshow(predicted_mask, cmap='gray')
        plt.title('Predicted Segmentation')
        plt.axis('off')

        # plot the overlayed images 
        plt.subplot(3, 5, i + 11)
        plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
        plt.title('Overlayed Image')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# load the previous saved model 
model_path = os.path.join(os.path.expanduser("~/Desktop"), "Street_segmentation_model.h5")
base_path = '/Users/dianachabarek/Desktop/Diana ML/pics/pics1/train_extra'
predict_and_visualize(model_path, base_path)
