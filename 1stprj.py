import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.models import Model

model = MobileNetV2(weights="imagenet")
#displaying model summary
model.summary()


# Load an image (Replace 'D:\OneDrive\Pictures\Camera Roll\WIN_20250318_22_56_14_Pro.jpg' with your image path)
img_path = r"D:\OneDrive\Pictures\Camera Roll\WIN_20250318_22_56_14_Pro.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

# Preprocess the image for VGG16
input_img = np.expand_dims(img, axis=0)
input_img = preprocess_input(input_img)

# Show the original image
plt.imshow(img)
plt.axis("off")
plt.show()

#make predictions
# Predict class probabilities
preds = model.predict(input_img)
top_pred_idx = np.argmax(preds[0])

# Print the top predicted class
print("Predicted class:", decode_predictions(preds, top=1)[0])

def grad_cam(model, img_array, layer_name):
    # Create a model that maps the input image to activations of the layer and the output predictions
    grad_model = Model([model.input], [model.get_layer(layer_name).output, model.output])

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by its importance
    conv_output = conv_output.numpy()[0]
    heatmap = np.mean(conv_output * pooled_grads.numpy(), axis=-1)

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap
# Select a convolutional layer (try 'block5_conv3' in VGG16)
layer_name = "block5_conv3"

# Compute Grad-CAM
heatmap = grad_cam(model, input_img, layer_name)

# Display heatmap
plt.matshow(heatmap)
plt.colorbar()
plt.show()

def superimpose_heatmap(img, heatmap, alpha=0.5):
    # Resize heatmap to match the image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose heatmap onto image
    superimposed_img = cv2.addWeighted(img, alpha, heatmap, 1 - alpha, 0)

    return superimposed_img

# Apply heatmap on original image
superimposed_img = superimpose_heatmap(img, heatmap)

# Display the result
plt.imshow(superimposed_img)
plt.axis("off")
plt.show()
