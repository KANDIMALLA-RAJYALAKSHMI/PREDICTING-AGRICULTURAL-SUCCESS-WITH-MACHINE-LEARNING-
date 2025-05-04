# PREDICTING-AGRICULTURAL-SUCCESS-WITH-MACHINE-LEARNING

from flask import Flask, request, render_template
import numpy as np
import pickle

# Load ML model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Crop labels with corresponding YouTube video URLs
crop_dict = {
    1: ("Rice", "https://youtu.be/FW_bw9jdrlQ?si=dNUo1DDd8aK-gcmj"),
    2: ("Maize", "https://youtu.be/nfMLKP1nXK0?si=B-XsdpSPsJKLit_G"),
    3: ("Jute", "https://youtu.be/Q3LZT6WaySo?si=CJC2AEkkLk4B-53u"),
    4: ("Cotton", "https://youtu.be/eN-TqqBQOAk?si=cOfvY0YeN4ZjSpbT"),
    5: ("Coconut", "https://youtu.be/GjSGdqT3DME?si=O5BV7WRxY7u1O-1L"),
    6: ("Papaya", "https://youtu.be/ulpw9ytIgTU?si=_yqVXKhXfqYU6RRm"),
    7: ("Orange", "https://youtu.be/_KnbD2ni0PI?si=I8xzBkdBHAPzElAP"),
    8: ("Apple", "https://youtu.be/SSSJCroA3Zs?si=s0uil67pI2z5pZ0I"),
    9: ("Muskmelon", "https://www.youtube.com/watch?v=d8PO-HEMmRo"),
    10: ("Watermelon", "https://youtu.be/XQ-DhDzBJPQ?si=4pu2S492soK7VhYW"),
    11: ("Grapes", "https://youtu.be/8Ik7b6UcDP8?si=Xym2ClK90FREOyYL"),
    12: ("Mango", "https://youtu.be/0iAZa5bHQj0?si=dEGllHmkTEGI9uKG"),
    13: ("Banana", "https://youtu.be/jyOnH6blEOU?si=ymbBXT8yUgpv4KCj"),
    14: ("Pomegranate","https://youtu.be/GM9HUHaw8Wk?si=B9hh6Ua3s2qnRO1h"),
    15: ("Lentil", "https://youtu.be/2I_RLnlKZyk?si=ZTBcgsiADJIK4Of8"),
    16: ("Blackgram", "https://youtu.be/ZynFEdm9Knw?si=OoARxOdEkxUIJLIc"),
    17: ("Mungbean", "https://youtu.be/W1hUVi_4U4I?si=hH-JGI1iqXArlfgQ"),
    18: ("Mothbeans", "https://youtu.be/kcMikS-J7bw?si=yoC1Z0YxegcNlaLs"),
    19: ("Pigeonpeas", "https://youtu.be/M7CvnL88byo?si=3f4lCyYA860ATN5t"),
    20: ("Kidneybeans", "https://youtu.be/mqkj-_POaqU?si=UQkO2poYIdhh9tj8"),
    21: ("Chickpea", "https://youtu.be/tg7ycEGagJM?si=9V5JJ0Yr6mRozpJh"),
    22: ("Coffee", "https://youtu.be/vzGa9Wi-KwM?si=mVcZFJJSoTD7UcBJ")
}

app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['Nitrogen']),
            float(request.form['Phosporus']),
            float(request.form['Potassium']),
            float(request.form['Temperature']),
            float(request.form['Humidity']),
            float(request.form['pH']),
            float(request.form['Rainfall'])
        ]

        sample = np.array(features).reshape(1, -1)
        mx_scaled = mx.transform(sample)
        final_scaled = sc.transform(mx_scaled)
        prediction = model.predict(final_scaled)[0]

        if prediction in crop_dict:
            crop_name, video_url = crop_dict[prediction]
            result = f"{crop_name} is the best crop to be cultivated right there"
        else:
            result = "Sorry, we could not determine the best crop with the provided data."
            video_url = None

        return render_template('index.html', result=result, video_url=video_url)

    except Exception as e:
        return render_template('index.html', result="Error: " + str(e), video_url=None)

if __name__ == "__main__":
    app.run(debug=True)

#HTML CODE
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Crop Recommendation System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <style>
    h1 { color: black; text-align: center; }
    .card { margin-left: auto; margin-right: auto; margin-top: 20px; color: white; }
    .container { background: #fffbfe; font-weight: bold; padding-bottom: 10px; border-radius: 15px; }
    a { font-size: 18px; }
  </style>
  <body style="background:#BCBBB8">

<nav class="navbar navbar-expand-lg navbar-dark bg-dark">
  <div class="container-fluid">
    <a class="navbar-brand" href="/">Crop Recommendation System Using Machine Learning</a>
  </div>
</nav>

<div class="container my-3 mt-3">
  <h1 class="text-success">Crop Recommendation System<span class="text-success">🌱</span></h1>

  <form action="/predict" method="POST">
    <div class="row">
      <div class="col-md-4">
        <label for="Nitrogen">Nitrogen</label>
        <input type="number" name="Nitrogen" class="form-control" required>
      </div>
      <div class="col-md-4">
        <label for="Phosporus">Phosphorus</label>
        <input type="number" name="Phosporus" class="form-control" required>
      </div>
      <div class="col-md-4">
        <label for="Potassium">Potassium</label>
        <input type="number" name="Potassium" class="form-control" required>
      </div>
    </div>

    <div class="row mt-4">
      <div class="col-md-4">
        <label for="Temperature">Temperature (°C)</label>
        <input type="number" step="0.01" name="Temperature" class="form-control" required>
      </div>
      <div class="col-md-4">
        <label for="Humidity">Humidity (%)</label>
        <input type="number" step="0.01" name="Humidity" class="form-control" required>
      </div>
      <div class="col-md-4">
        <label for="pH"> Soil PH</label>
        <input type="number" step="0.01" name="pH" class="form-control" required>
      </div>
    </div>

    <div class="row mt-4">
      <div class="col-md-4">
        <label for="Rainfall">Rainfall (mm)</label>
        <input type="number" step="0.01" name="Rainfall" class="form-control" required>
      </div>
    </div>

    <div class="row mt-4">
      <div class="col-md-12 text-center">
        <button type="submit" class="btn btn-primary btn-lg">Get Recommendation</button>
      </div>
    </div>
  </form>

  {% if result %}
    <div class="card bg-dark" style="width: 24rem;">
      <img src="{{url_for('static', filename='crop.png')}}" class="card-img-top mx-auto mt-3" style="width: 70px; height: 70px;" alt="Crop">
      <div class="card-body text-center">
        <h5 class="card-title">Recommended Crop:</h5>
        <p class="card-text">{{ result }}</p>
      </div>
    </div>
  {% endif %}

  {% if video_url %}
    <div class="text-center mt-3">
      <a href="{{ video_url }}" target="_blank">Click here to watch related video on YouTube</a>
    </div>
  {% endif %}
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
