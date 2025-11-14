from flask import Flask, jsonify, request
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas

# JSON estático inicial
data_cubo = {
    "timestamp": int(time.time() * 1000),  # milisegundos desde epoch
    "x": 0.25,
    "y": 1.10,
    "z": 1,
    "pitch": 15.0,
    "yaw": 45.0,
    "roll": 5.0
}

@app.route("/api/posicion", methods=["GET"])
def get_posicion():
    return jsonify(data_cubo)

@app.route("/api/posicion", methods=["POST"])
def update_posicion():
    global data_cubo
    nueva_data = request.json

    if not nueva_data:
        return jsonify({"error": "No se recibieron datos"}), 400

    # Se asegura de incluir timestamp en milisegundos
    nueva_data["timestamp"] = int(time.time() * 1000)
    data_cubo = nueva_data

    return jsonify({"status": "OK", "data": data_cubo}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
