import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import cv2
import os

# ✅ Load the trained model
model = load_model("emotion_vgg16.h5")  # Change filename if needed
print("✅ Model Loaded Successfully!")

# ✅ Check model summary to verify convolutional layers
model.summary()

# ✅ Extract the VGG16 model from the Sequential model
vgg16_model = model.get_layer("vgg16")

# ✅ Find last convolutional layer in VGG16
conv_layers = [layer.name for layer in vgg16_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
if not conv_layers:
    raise ValueError("❌ No convolutional layer found in VGG16 model.")

last_conv_layer = vgg16_model.get_layer(conv_layers[-1])  # Get the last Conv2D layer
print(f"✅ Using last Conv2D layer: {last_conv_layer.name}")

# ✅ Set image path (Change to your actual image path)
img_path = r"D:\OneDrive\Desktop\test.jpg"
if not os.path.exists(img_path):
    raise FileNotFoundError(f"❌ Image not found: {img_path}")

# ✅ Load and preprocess image (Match VGG16 input size: 224x224)
img = image.load_img(img_path, target_size=(224, 224))  
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

# ✅ Create Grad-CAM model
grad_model = Model(inputs=vgg16_model.input, outputs=[last_conv_layer.output, vgg16_model.output])


# ✅ Compute gradients
with tf.GradientTape() as tape:
    conv_output, predictions = grad_model(img_array)
    class_channel = tf.reduce_max(predictions)  # Take max response

# ✅ Get gradients of the output w.r.t conv output
grads = tape.gradient(class_channel, conv_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# ✅ Multiply each feature map by importance weight
heatmap = np.mean(conv_output[0] * pooled_grads, axis=-1)

# ✅ Normalize heatmap
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)

# ✅ Resize heatmap to match original image size
heatmap = cv2.resize(heatmap, (224, 224))

# ✅ Apply colormap
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# ✅ Superimpose heatmap on image
original = cv2.imread(img_path)
original = cv2.resize(original, (224, 224))
superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# ✅ Display activation map
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image.load_img(img_path))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
plt.title("Activation Map")
plt.axis("off")

plt.show()
