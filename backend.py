import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# =========================
# CONFIG
# =========================
Image_Size = 128
class_labels = ['glioma','meningioma','notumor','pituitary']


# =========================
# LOAD MODEL (CACHED)
# =========================
import gdown
url = "https://drive.google.com/uc?id=1zBD2OdSzpKPlhpXANJUiOW0p7_IdhRnt"
output = "model.h5"
gdown.download(url, output, quiet=False)
@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()


# =========================
# GRAD CAM
# =========================
def grad_cam(img_array, model, last_conv_layer_name="block5_conv3"):

    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    backbone = tf.keras.Model(base_model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input

    layer_index = base_model.layers.index(last_conv_layer)
    for layer in base_model.layers[layer_index + 1:]:
        x = layer(x)

    for layer in model.layers[1:]:
        x = layer(x)

    classifier = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        conv_outputs = backbone(img_array)
        tape.watch(conv_outputs)
        predictions = classifier(conv_outputs)

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = heatmap.numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8

    return heatmap


# =========================
# PREDICT FUNCTION
# =========================
def predict(image):

    img = image.resize((Image_Size, Image_Size))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    preds = model.predict(img_array)
    pred_class = np.argmax(preds)
    confidence = np.max(preds)
    class_name = class_labels[pred_class]

    heatmap = grad_cam(img_array, model)

    heatmap = cv2.resize(heatmap, (Image_Size, Image_Size))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.resize(np.array(image), (Image_Size, Image_Size))
    superimposed = heatmap * 0.4 + original

    if class_name == "notumor":
        result = "No Tumor Detected"
    else:
        result = f"Tumor Detected: {class_name}"

    return result, confidence, superimposed.astype("uint8")


# =========================
# STREAMLIT UI
# =========================
st.title("Brain Tumor Classifier + GradCAM")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")


    if st.button("Predict"):

        with st.spinner("Analyzing MRI..."):
            result, confidence, cam = predict(image)

        st.success(result)
        st.write(f"Confidence: {confidence*100:.2f}%")
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            st.image(cam, caption="Grad-CAM Visualization", use_container_width=True)
