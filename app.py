import os
import json
import numpy as np
import pickle
from datetime import datetime
from flask import Flask, request, render_template, jsonify
from dotenv import load_dotenv
import requests
from pytz import timezone  # ✅ Added for IST support

# Load environment variables
load_dotenv('key.env')

TOGETHER_API = "https://api.together.xyz/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {os.getenv('TOGETHER_API_KEY')}",
    "Content-Type": "application/json"
}

app = Flask(__name__)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# ✅ Load model and scaler
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        holiday = int(request.form['holiday'])
        weather = int(request.form['weather'])
        temp = float(request.form['temp'])
        rain = int(request.form['rain'])
        snow = int(request.form['snow'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hour = int(request.form['hours'])
        minute = int(request.form['minutes'])
        second = int(request.form['seconds'])

        features = np.array([[holiday, temp, rain, snow, weather, year, month, day, hour, minute, second]])
        scaled = scaler.transform(features)
        prediction = int(model.predict(scaled)[0])

        traffic_labels = [f"{i:02}:00" for i in range(24)]
        traffic_data = [np.random.randint(300, 600) for _ in range(24)]
        traffic_data[hour] = prediction
        avg_data = [int(prediction * 0.75)] * 24

        # ✅ Get current time in IST
        ist = timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        timestamp = now_ist.strftime("%d/%m/%Y, %I:%M:%S %p")

        return render_template("result.html",
            predicted_volume=prediction,
            confidence=94,
            peak_hour="18:00",
            peak_volume=prediction + 500,
            rush_hour_count=2,
            current_volume=prediction + 100,
            daily_average=int(prediction * 0.75),
            min_volume=int(prediction * 0.25),
            min_hour="04:00",
            rush_periods="6:00–7:00 PM",
            timestamp=timestamp,
            labels=traffic_labels,
            data=traffic_data,
            avg_data=avg_data,
            current_index=hour
        )
    except Exception as e:
        return f"❌ Error: {e}"

@app.route("/api/chat", methods=["POST"])
def chat_api():
    user_msg = request.json.get("message", "")
    payload = {
        "model": "meta-llama/Llama-3-8b-chat-hf",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You're 🚦 TrafficBot — a friendly, knowledgeable assistant in the TrafficTelligence web platform. "
                    "Your job is to help users understand traffic patterns, predictions, and platform features. "
                    "Speak in a casual, cheerful tone — use emojis and simple language.\n\n"
                    "About TrafficTelligence:\n"
                    "- Predicts traffic volumes using ML based on weather, date/time, rain/snow, and holidays.\n"
                    "- Built with Python (Flask), Tailwind CSS, and Chart.js.\n"
                    "- Developed by 👨‍💻 Medisetty Shanmukha Sri Saikumar → https://www.linkedin.com/in/medisetty-shanmukha-sri-saikumar-b6195731a\n\n"
                    "💡 Tip: You can understand multiple languages like Hindi, Telugu, and Tamil — respond in the user's language if detected.\n"
                    "Keep replies short, helpful, and friendly 🙂"
                )
            },
            {"role": "user", "content": user_msg}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
        "stream": False
    }

    upstream = requests.post(TOGETHER_API, headers=HEADERS, data=json.dumps(payload), timeout=30)

    if upstream.status_code != 200:
        return jsonify({"reply": "❗ Together AI error."}), upstream.status_code

    result = upstream.json()
    reply = result["choices"][0]["message"]["content"]
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)