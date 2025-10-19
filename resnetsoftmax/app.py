"""
app.py for ResNet-18 + Softmax Classifier

- Loads the pre-trained ResNet-18 model for feature extraction.
- Loads the trained Softmax classifier (W, b, mean, std) from 'softmax_model_resnet.pkl'.
- Provides a Streamlit interface for users to upload an image and get a prediction.
- This app is specifically designed to work with the model trained by 'feature_2_resnet.py'.
"""

import streamlit as st
import pickle
import numpy as np
import os
import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# =======================
# 1. Cấu hình & Tải mô hình
# =======================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- SỬA LỖI: Sử dụng đường dẫn tuyệt đối để đảm bảo luôn tìm thấy tệp ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "outputs", "softmax_model_resnet.pkl")


@st.cache_resource
def load_models():
    """Tải mô hình ResNet-18 và mô hình Softmax đã huấn luyện."""
    # 1. Tải mô hình ResNet-18 để trích xuất đặc trưng
    weights = ResNet18_Weights.IMAGENET1K_V1
    transform_for_resnet = weights.transforms()
    resnet_model = resnet18(weights=weights).to(DEVICE)

    # Loại bỏ lớp phân loại cuối cùng của ResNet để lấy đặc trưng
    resnet_model.fc = torch.nn.Identity()
    resnet_model.eval()

    # 2. Tải mô hình Softmax đã huấn luyện
    if not os.path.exists(MODEL_PATH):
        st.error(f"Lỗi: Không tìm thấy tệp mô hình tại '{MODEL_PATH}'.")
        st.info("Hãy chắc chắn rằng bạn đã chạy tệp 'feature_2_resnet.py' thành công.")
        return None, None, None, None, None, None, None

    with open(MODEL_PATH, "rb") as f:
        softmax_model = pickle.load(f)

    W = softmax_model["W"]
    b = softmax_model["b"]
    feature_mean = softmax_model["mean"]
    feature_std = softmax_model["std"]

    # --- SỬA LỖI KeyError: Tạo một bản đồ ngược để tra cứu tên nhãn từ chỉ số ---
    label_map = softmax_model["label_map"]
    # original label_map: {"non-phone": 0, "defective": 1, ...}
    # reversed_label_map: {0: "non-phone", 1: "defective", ...}
    reversed_label_map = {v: k for k, v in label_map.items()}

    return resnet_model, transform_for_resnet, W, b, feature_mean, feature_std, reversed_label_map


# Tải tất cả các mô hình và dữ liệu cần thiết
resnet_model, transform_for_resnet, W, b, feature_mean, feature_std, reversed_label_map = load_models()


# =======================
# 2. Các hàm xử lý
# =======================
def extract_resnet_features(pil_image, model, transform, device):
    """Trích xuất vector đặc trưng từ ảnh PIL bằng ResNet-18."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    img_t = transform(pil_image)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        features = model(batch_t)

    return features.cpu().numpy()


def softmax(z):
    """Tính toán hàm softmax."""
    if z.ndim == 1:
        z = z.reshape(1, -1)
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def predict(X_feature):
    """Dự đoán từ vector đặc trưng đã chuẩn hóa."""
    scores = X_feature @ W + b
    probs = softmax(scores)
    preds = np.argmax(probs, axis=1)
    return preds, probs


# =======================
# 3. Giao diện Streamlit
# =======================
st.title("Phân loại ảnh điện thoại (ResNet-18 + Softmax)")
st.write(
    "Ứng dụng này sử dụng mô hình ResNet-18 để trích xuất đặc trưng và một bộ phân loại Softmax để dự đoán trạng thái của điện thoại.")

if resnet_model is None:
    st.stop()

uploaded_file = st.file_uploader("Tải lên một ảnh để dự đoán", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh bạn đã tải lên", use_column_width=True)

    with st.spinner('Đang phân tích ảnh... (sử dụng GPU nếu có)'):
        # 1. Trích xuất đặc trưng ResNet
        resnet_features = extract_resnet_features(image, resnet_model, transform_for_resnet, DEVICE)

        # 2. Chuẩn hóa đặc trưng
        standardized_features = (resnet_features - feature_mean) / feature_std

        # 3. Dự đoán
        pred_class, probs = predict(standardized_features)

        st.write("---")
        st.subheader("🎉 Kết quả dự đoán")

        predicted_label_index = pred_class[0]
        predicted_label_name = reversed_label_map.get(predicted_label_index, "Không xác định")

        if predicted_label_name == "defective":
            st.error(f"👉 Dự đoán: {predicted_label_name.upper()} (Lỗi)")
        elif predicted_label_name == "non-defective":
            st.success(f"👉 Dự đoán: {predicted_label_name.upper()} (Không lỗi)")
        else:
            st.warning(f"👉 Dự đoán: {predicted_label_name.upper()}")

        st.subheader("📊 Xác suất dự đoán cho từng lớp:")
        for i in range(len(reversed_label_map)):
            class_name = reversed_label_map[i]
            st.write(f"- **{class_name}**: `{probs[0][i]:.4f}`")