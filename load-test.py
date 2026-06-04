import threading
import urllib.request
import urllib.parse
import time
import os

URL = "http://127.0.0.1:8080/predict"
IMAGE_PATH = "test-skin.jpg"
TOTAL_REQUESTS = 50   # Lebih sedikit karena /predict lebih berat
CONCURRENT = 5

print(f"🚀 Mulai load test ke /predict")
print(f"📡 {TOTAL_REQUESTS} requests, {CONCURRENT} concurrent")
print("-" * 50)

counter = {"success": 0, "error": 0}
lock = threading.Lock()

def send_request():
    try:
        with open(IMAGE_PATH, "rb") as f:
            image_data = f.read()

        boundary = "----FormBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="test.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode() + image_data + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            URL,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        urllib.request.urlopen(req, timeout=100)
        with lock:
            counter["success"] += 1
    except Exception as e:
        with lock:
            counter["error"] += 1
            print(f"ERROR: {e}")

start = time.time()

for batch in range(TOTAL_REQUESTS // CONCURRENT):
    threads = []
    for _ in range(CONCURRENT):
        t = threading.Thread(target=send_request)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    elapsed = time.time() - start
    print(f"Batch {batch+1}: ✅ {counter['success']} sukses | ❌ {counter['error']} error | ⏱️ {elapsed:.1f}s")

print("-" * 50)
print(f"✅ Selesai! Sukses: {counter['success']}, Error: {counter['error']}")