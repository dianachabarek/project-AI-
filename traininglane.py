# import necessary libraries 
import os
import cv2
import glob
import numpy as np
import json
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# function for loading the images 
# resizing the resolution (256 x 512) , because the images are too big (2048 x 1024)
def load_images_and_segmentations(base_path, pic_subfolder, seg_subfolder, target_size=(256, 512)):  
    images = []
    segmentations = []
    image_paths = glob.glob(os.path.join(base_path, pic_subfolder, '**', '*.png'), recursive=True)

    for img_file in image_paths:
        # if the image does not load, give me an error  
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Could not read image {img_file}")
            continue
        # resize and append it to the list 
        img = cv2.resize(img, target_size)
        images.append(img)

        
        city_name = os.path.basename(os.path.dirname(img_file))
        base_filename = os.path.splitext(os.path.basename(img_file))[0].replace('_leftImg8bit', '')
        json_file_name = f"{base_filename}_gtCoarse_polygons.json"
        json_file_path = os.path.join(base_path, seg_subfolder, city_name, json_file_name)

        # prove if the path exist and json files, if not give me an error 
        if os.path.exists(json_file_path):
            # open the json file in read format 
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)
                # json files into image masks, training images with images instead images with texts 
                mask = Image.new('L', target_size, 0)
                draw = ImageDraw.Draw(mask)
                for obj in data['objects']:
                    scaled_polygon = [(x[0] * target_size[0] / data['imgWidth'], x[1] * target_size[1] / data['imgHeight']) for x in obj['polygon']]
                    draw.polygon(scaled_polygon, outline=1, fill=1)
                mask_np = np.array(mask)
                segmentations.append(mask_np)
        else:
            print(f"JSON file not found for {img_file}")
    # return the successfully loaded images and the drawn images of json files
    return np.array(images), np.array(segmentations)

# CNN, activate it with relu 
def simple_unet_model(input_size=(512, 256, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    # upsampling with transpose convolution 
    u5 = UpSampling2D((2, 2))(c4) 
    c5 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u5)
    u6 = UpSampling2D((2, 2))(c5)
    c6 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    u7 = UpSampling2D((2, 2))(c6)
    c7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    # values between 0 and 1 
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c7)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# set base_path to your actual data location
base_path = "/Users/dianachabarek/Desktop/Diana ML"
images, masks = load_images_and_segmentations(base_path, 'pics/pics1/train_extra', 'segm/segreal/train_extra')

# split the data in train, val, test
# 70% train, 15% test , 15% val 
X_train, X_temp, y_train, y_temp = train_test_split(images, masks, test_size=0.15, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# object creation 
model = simple_unet_model()

# set the learning rate, avoiding over/underfitting
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# training with GPU 
with tf.device('/device:GPU:0'):
    # set paarmeters for training: batch size and epochs 
    model.fit(X_train, y_train, batch_size=1, epochs=5, validation_data=(X_val, y_val), verbose=1)

# save the model 
model_save_path = os.path.join(os.path.expanduser("~/Desktop"), "Street_segmentation_model.h5")
model.save(model_save_path)
print(f"Model saved at {model_save_path}")
