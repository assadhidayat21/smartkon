from flask import Flask, request, render_template
import joblib
import numpy as np
import mysql.connector


app = Flask(__name__)

# Database Connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="smartkon"
)
cursor = db.cursor()


# Load the models
model_c4_5 = joblib.load('models/c45_model.pkl')
model_rf = joblib.load('models/rf_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/history')
def history():
    cursor.execute("SELECT * FROM hasil_prediksi ORDER BY waktu DESC")
    hasil = cursor.fetchall()
    return render_template('history.html', data=hasil)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        usia_istri = int(request.form['usia_istri'])
        pendidikan_istri = request.form['pendidikan_istri']
        pendidikan_suami = request.form['pendidikan_suami']
        jumlah_anak = int(request.form['jumlah_anak'])
        agama = request.form['agama']
        istri_bekerja = request.form['istri_bekerja']
        kesibukan_suami = request.form['kesibukan_suami']
        standar_hidup = request.form['standar_hidup']
        ekspose_media = request.form['ekspose_media']
        model_choice = request.form['model_choice']

        # Mengonversi data kategorikal menjadi numerik
        pendidikan_mapping = {'SD': 0, 'SMP': 1, 'SMA': 2, 'Sarjana': 3}
        agama_mapping = {'Islam': 0, 'Non-Islam': 1}
        istri_bekerja_mapping = {'Ya': 1, 'Tidak': 0}
        kesibukan_mapping = {'rendah': 0, 'sedang': 1, 'tinggi': 2, 'sangat tinggi': 3}
        standar_hidup_mapping = {'rendah': 0, 'sedang': 1, 'tinggi': 2, 'sangat tinggi': 3}
        ekspose_mapping = {'Ya': 1, 'Tidak': 0}

        # Mengubah input menjadi format numerik
        input_data = [
            usia_istri,
            pendidikan_mapping[pendidikan_istri],
            pendidikan_mapping[pendidikan_suami],
            jumlah_anak,
            agama_mapping[agama],
            istri_bekerja_mapping[istri_bekerja],
            kesibukan_mapping[kesibukan_suami],
            standar_hidup_mapping[standar_hidup],
            ekspose_mapping[ekspose_media]
        ]

        # Melakukan prediksi berdasarkan pilihan model
        if model_choice == 'C4.5':
            prediction = model_c4_5.predict([input_data])[0]
            proba = model_c4_5.predict_proba([input_data])[0]
        elif model_choice == 'Random Forest':
            prediction = model_rf.predict([input_data])[0]
            proba = model_rf.predict_proba([input_data])[0]

        # Menyiapkan hasil prediksi dan probabilitas
        if prediction == 0:
            prediction_text = "No-Use"
            prob_text = f"{proba[0]*100:.2f}"
        elif prediction == 1:
            prediction_text = "Short-Term"
            prob_text = f"{proba[1]*100:.2f}"
        elif prediction == 2:
            prediction_text = "Long-Term"
            prob_text = f"{proba[2]*100:.2f}"

        # Simpan ke database
        sql = """
            INSERT INTO hasil_prediksi (
                usia_istri, pendidikan_istri, pendidikan_suami, jumlah_anak,
                agama, istri_bekerja, kesibukan_suami, standar_hidup,
                ekspose_media, model_choice, hasil_prediksi, probabilitas
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        val = (
            usia_istri,
            pendidikan_istri,
            pendidikan_suami,
            jumlah_anak,
            agama,
            istri_bekerja,
            kesibukan_suami,
            standar_hidup,
            ekspose_media,
            model_choice,
            prediction_text,
            prob_text
        )
        cursor.execute(sql, val)
        db.commit()

        return render_template('index.html', prediction_text=prediction_text, prob_text=prob_text)

    except ValueError as e:
        return render_template('index.html', prediction_text="Error: Pastikan semua input adalah angka yang valid.")


if __name__ == "__main__":
    app.run(debug=True)
