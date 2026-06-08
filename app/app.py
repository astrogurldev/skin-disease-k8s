import os
import io
from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch
from prometheus_flask_exporter import PrometheusMetrics
from disease_info import get_disease_info

app = Flask(__name__)
metrics = PrometheusMetrics(app)

print("🔄 Loading model...")
MODEL_NAME = "Jayanth2002/dinov2-base-finetuned-SkinDisease"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
print("✅ Model loaded!")

CLASS_NAMES = [
    'Basal Cell Carcinoma', 'Darier\'s Disease', 'Epidermolysis Bullosa Pruriginosa',
    'Hailey-Hailey Disease', 'Herpes Simplex', 'Impetigo', 'Larva Migrans',
    'Leprosy Borderline', 'Leprosy Lepromatous', 'Leprosy Tuberculoid',
    'Lichen Planus', 'Lupus Erythematosus Chronicus Discoides', 'Melanoma',
    'Molluscum Contagiosum', 'Mycosis Fungoides', 'Neurofibromatosis',
    'Papilomatosis Confluentes And Reticulate', 'Pediculosis Capitis',
    'Pityriasis Rosea', 'Porokeratosis Actinic', 'Psoriasis',
    'Tinea Corporis', 'Tinea Nigra', 'Tungiasis', 'Actinic Keratosis',
    'Dermatofibroma', 'Nevus', 'Pigmented Benign Keratosis',
    'Seborrheic Keratosis', 'Squamous Cell Carcinoma', 'Vascular Lesion'
]

@app.route("/")
def index():
    pod_name = os.environ.get("POD_NAME", "local-dev")
    return render_template("index.html", pod_name=pod_name)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang dikirim"}), 400

    file = request.files["file"]
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
    top3 = torch.topk(probs, 3).indices.tolist()

    results = []
    for idx in top3:
        name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class {idx}"
        confidence = probs[idx].item() * 100
        info = get_disease_info(name)
        results.append({
            "disease": name,
            "confidence": round(confidence, 1),
            "info": info
        })

    return jsonify({
        "pod_name": os.environ.get("POD_NAME", "local-dev"),
        "results": results
    })

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)