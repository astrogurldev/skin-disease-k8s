# ============================================
# BAGIAN 1: Import library yang dibutuhkan
# ============================================
import os
import io
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
from prometheus_flask_exporter import PrometheusMetrics

# ============================================
# BAGIAN 2: Inisialisasi Flask app
# ============================================
app = Flask(__name__)
metrics = PrometheusMetrics(app)  # untuk monitoring nanti

# ============================================
# BAGIAN 3: Load model dari HuggingFace
# ============================================
print("🔄 Loading model... (ini butuh waktu pertama kali)")

MODEL_NAME = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()  # mode inference, bukan training

print("✅ Model loaded!")

CLASS_NAMES = [
    'Basal Cell Carcinoma',
    'Darier\'s Disease',
    'Epidermolysis Bullosa Pruriginosa',
    'Hailey-Hailey Disease',
    'Herpes Simplex',
    'Impetigo',
    'Larva Migrans',
    'Leprosy Borderline',
    'Leprosy Lepromatous',
    'Leprosy Tuberculoid',
    'Lichen Planus',
    'Lupus Erythematosus Chronicus Discoides',
    'Melanoma',
    'Molluscum Contagiosum',
    'Mycosis Fungoides',
    'Neurofibromatosis',
    'Papilomatosis Confluentes And Reticulate',
    'Pediculosis Capitis',
    'Pityriasis Rosea',
    'Porokeratosis Actinic',
    'Psoriasis',
    'Tinea Corporis',
    'Tinea Nigra',
    'Tungiasis',
    'Actinic Keratosis',
    'Dermatofibroma',
    'Nevus',
    'Pigmented Benign Keratosis',
    'Seborrheic Keratosis',
    'Squamous Cell Carcinoma',
    'Vascular Lesion'
]

# ============================================
# BAGIAN 4: Routes (halaman/endpoint)
# ============================================

# Route 1: Halaman utama (tampilan web)
@app.route("/")
def index():
    # Ambil nama pod dari environment variable Kubernetes
    pod_name = os.environ.get("POD_NAME", "local-dev")
    return render_template("index.html", pod_name=pod_name)


# Route 2: API endpoint untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    # Cek apakah ada file yang dikirim
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400

    file = request.files["file"]

    # Baca gambar
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Proses gambar untuk model
    inputs = processor(images=image, return_tensors="pt")

    # Prediksi
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Ambil hasil
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    top3_indices = torch.topk(probabilities, 3).indices.tolist()

    results = []
    for idx in top3_indices:
        results.append({
            "disease": CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}",
            "confidence": f"{probabilities[idx].item() * 100:.1f}%"
        })

    pod_name = os.environ.get("POD_NAME", "local-dev")

    return jsonify({
        "pod_name": pod_name,       # biar keliatan dari pod mana
        "results": results
    })


# Route 3: Health check (dipakai Kubernetes untuk cek apakah pod sehat)
@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200


# ============================================
# BAGIAN 5: Jalankan app
# ============================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)