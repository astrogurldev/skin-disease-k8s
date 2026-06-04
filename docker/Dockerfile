FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from transformers import AutoModelForImageClassification, AutoImageProcessor; AutoImageProcessor.from_pretrained('Jayanth2002/dinov2-base-finetuned-SkinDisease'); AutoModelForImageClassification.from_pretrained('Jayanth2002/dinov2-base-finetuned-SkinDisease'); print('Model cached!')"

COPY app/ .

EXPOSE 5000

CMD ["python", "app.py"]