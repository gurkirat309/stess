from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/data', methods=['POST'])
def get_data():
    data = request.json

    print("Received:", data)

    # Save to JSON file
    with open("wifi_data.json", "a") as f:
        f.write(json.dumps(data) + "\n")

    return "OK"

app.run(host="0.0.0.0",port=5000)