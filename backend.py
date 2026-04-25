import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing.image import load_img
import gdown
import os
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMAGE_SIZE   = 128
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]
# ─────────────────────────────────────────────
# MODEL  (loaded once, reused everywhere)
# ─────────────────────────────────────────────
_model = None
MODEL_PATH = "model.keras"
FILE_ID = "1YKAdM5FpywFf6LU_749-fGKNXN6NvKV7"
def download_model():
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)  # 🔥 force overwrite

    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

def get_model():
    global _model
    if _model is None:
        download_model()

        try:
            _model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                safe_mode=False
            )
        except Exception:
            _model = tf.keras.models.load_model(
                MODEL_PATH,
                compile=False,
                custom_objects={
                    "InputLayer": tf.keras.layers.InputLayer
                }
            )

        # warm-up
        _model.predict(np.zeros((1, 128, 128, 3)))

    return _model
# ─────────────────────────────────────────────
# GRAD-CAM++
# ─────────────────────────────────────────────
def grad_cam_plus_plus(img_array: np.ndarray,
                       model,
                       last_conv_layer_name: str = "block5_conv3",
                       class_idx: int = None) -> np.ndarray:
    """
    Grad-CAM++ using a forward hook — zero graph surgery, works with any
    Keras model structure (flat, nested, functional, sequential).
    img_array : float32 ndarray  (1, H, W, 3)  values in [0, 1]
    Returns   : float32 heatmap  (H_conv, W_conv) normalised to [0, 1]
    """

    # ── 1. find the target layer anywhere in the model tree ───────────
    def find_layer(m, name):
        for layer in m.layers:
            if layer.name == name:
                return layer
            if hasattr(layer, 'layers'):
                hit = find_layer(layer, name)
                if hit is not None:
                    return hit
        return None

    target_layer = find_layer(model, last_conv_layer_name)
    if target_layer is None:
        names = []
        def collect(m):
            for l in m.layers:
                names.append(f"{l.name}  ({type(l).__name__})")
                if hasattr(l, 'layers'):
                    collect(l)
        collect(model)
        raise ValueError(
            f"Layer '{last_conv_layer_name}' not found.\n"
            "Available layers:\n" + "\n".join(names)
        )

    # ── 2. intercept conv output with a callback layer ─────────────────
    # Monkey-patch the layer's __call__ to store its output in a list.
    # This requires NO graph rewiring — it works at the Python call level.
    conv_output_holder = []
    original_call = target_layer.__class__.call

    def hooked_call(self, *args, **kwargs):
        result = original_call(self, *args, **kwargs)
        if self is target_layer:
            conv_output_holder.append(result)
        return result

    target_layer.__class__.call = hooked_call

    # ── 3. run forward pass with GradientTape ─────────────────────────
    img_tensor = tf.cast(img_array, tf.float32)
    try:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv_output_holder.clear()
                preds = model(img_tensor, training=False)

                if not conv_output_holder:
                    raise RuntimeError(
                        f"Hook did not fire — layer '{last_conv_layer_name}' "
                        "was not called during the forward pass."
                    )

                conv_out = conv_output_holder[0]
                tape1.watch(conv_out)
                tape2.watch(conv_out)

                if class_idx is None:
                    class_idx = int(tf.argmax(preds[0]))
                score = preds[:, class_idx]

            g1 = tape1.gradient(score, conv_out)
        g2 = tape2.gradient(g1, conv_out)

    finally:
        # always restore the original method — even if an error occurs
        target_layer.__class__.call = original_call

    # ── 4. Grad-CAM++ weighting ────────────────────────────────────────
    alpha = g2 / (
        2.0 * g2
        + tf.reduce_sum(conv_out * g2 * g1, axis=(1, 2), keepdims=True)
        + 1e-7
    )
    weights = tf.reduce_sum(alpha * tf.nn.relu(g1), axis=(1, 2))

    cam = tf.reduce_sum(
        conv_out[0] * weights[0][tf.newaxis, tf.newaxis, :], axis=-1
    )
    cam = tf.nn.relu(cam).numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam.astype(np.float32)
# ─────────────────────────────────────────────
# LIME
# ─────────────────────────────────────────────


def explain_with_lime_overlay(img_rgb: np.ndarray, model):
    """
    LIME explanation using your exact logic (no augmentation).
    Input: img_rgb (H, W, 3) uint8
    """

    def lime_predict(images):
        images = np.array(images).astype(np.float32) / 255.0
        return model.predict(images, verbose=0)

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        img_rgb.astype(np.float64),
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    # cleaner boundaries
    lime_overlay = mark_boundaries(temp / 255.0, mask, color=(0, 1, 0))
    lime_overlay = (lime_overlay * 255).astype(np.uint8)

    return lime_overlay
# ─────────────────────────────────────────────
# OVERLAY HELPERS
# ─────────────────────────────────────────────
def _build_cam_overlay(img_rgb: np.ndarray,
                       heatmap: np.ndarray,
                       alpha: float = 0.45) -> np.ndarray:
    """Resize heatmap → JET colormap → blend with img_rgb. Returns uint8 RGB."""
    hm_up  = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    hm_col = cv2.applyColorMap(np.uint8(255 * hm_up), cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_col, cv2.COLOR_BGR2RGB)
    return np.clip(hm_rgb * alpha + img_rgb, 0, 255).astype(np.uint8)


def _build_fused_overlay(img_rgb: np.ndarray,
                         cam_overlay: np.ndarray,
                         lime_overlay: np.ndarray) -> np.ndarray:
    """Weighted blend: original 35 % + CAM++ 30 % + LIME 35 %."""
    fused = (img_rgb     * 0.35
           + cam_overlay  * 0.30
           + lime_overlay * 0.35)
    return np.clip(fused, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# PREDICT  (public API)
# ─────────────────────────────────────────────
def predict(image,
            model=None,
            last_conv_layer: str = "block5_conv3",
            lime_samples: int = 1000,
            lime_features: int = 10,
            run_lime_explain: bool = True):
    """
    Run inference + Grad-CAM++ + (optionally) LIME on a PIL RGB image.

    Parameters
    ----------
    image            : PIL.Image  (any size, RGB)
    model            : Keras model  (optional – uses cached model if None)
    last_conv_layer  : name of the last conv layer for CAM
    lime_samples     : number of LIME perturbations  (higher = smoother)
    lime_features    : number of LIME superpixels to highlight
    run_lime_explain : set False to skip LIME (much faster)

    Returns
    -------
    result        : str    human-readable verdict
    confidence    : float  probability of predicted class  (0–1)
    cam_overlay   : uint8 ndarray (IMAGE_SIZE, IMAGE_SIZE, 3)  Grad-CAM++ blend
    lime_overlay  : uint8 ndarray (IMAGE_SIZE, IMAGE_SIZE, 3)  LIME boundaries
                    (None if run_lime_explain=False)
    fused_overlay : uint8 ndarray (IMAGE_SIZE, IMAGE_SIZE, 3)  triple-fused view
                    (None if run_lime_explain=False)
    pred_class    : int    index into CLASS_LABELS
    preds         : ndarray (num_classes,)  all class probabilities
    """
    if model is None:
        model = get_model()

    # ── pre-process ──────────────────────────────────────────
    img       = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_rgb   = np.array(img, dtype=np.uint8)          # (H,W,3) uint8 for LIME / blending
    img_array = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)

    # ── inference ────────────────────────────────────────────
    preds      = model.predict(img_array, verbose=0)[0]   # (num_classes,)
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))
    class_name = CLASS_LABELS[pred_class]

    # ── Grad-CAM++ overlay ───────────────────────────────────
    heatmap     = grad_cam_plus_plus(img_array, model, last_conv_layer, pred_class)
    cam_overlay = _build_cam_overlay(img_rgb, heatmap)

    # ── LIME overlay (optional) ──────────────────────────────
    lime_overlay  = None
    fused_overlay = None
    if run_lime_explain:
        lime_overlay = explain_with_lime_overlay(img_rgb, model)
        fused_overlay = _build_fused_overlay(img_rgb, cam_overlay, lime_overlay)

    result = (
        "No Tumor Detected"
        if class_name == "notumor"
        else f"Tumor Detected: {class_name.capitalize()}"
    )

    return result, confidence, cam_overlay, lime_overlay, fused_overlay, pred_class, preds