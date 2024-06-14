from flask import Flask, render_template, request, send_file, jsonify, redirect, url_for, flash, session
from wtforms import Form, StringField, PasswordField, SubmitField, validators

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import load_model, Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from flask import Flask, render_template, request, send_file, jsonify
import json
from datetime import datetime, timedelta
import argparse
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
DEVELOPMENT_ENV = True

app_data = {
    "name": "Peter's Starter Template for a Flask Web App",
    "description": "A basic Flask app using bootstrap for layout",
    "author": "Komando Lukman Sucipto",
    "html_title": "SIPEKAN",
    "project_name": "SIPEKAN",
    "keywords": "flask, webapp, template, basic, Prediksi, LSTM",
}

app.secret_key = 'your_secret_key'  # Change this to a secret key for your application

# Sample user data (replace with your user authentication logic)
users = {'admin': {'password': 'admin'}}

# Login form using WTForms
class LoginForm(Form):
    username = StringField('Username', [validators.InputRequired()])
    password = PasswordField('Password', [validators.InputRequired()])
    submit = SubmitField('Login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        username = form.username.data
        password = form.password.data
        if username in users and users[username]['password'] == password:
            session['username'] = username  # Store the username in the session
            flash('Login successful', 'success')
            return redirect(url_for('index'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')
    return render_template('login.html', app_data=app_data, form=form)

@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from the session
    return redirect(url_for('login'))

@app.route('/')
def index():
    if 'username' in session:
        return render_template('index.html', app_data=app_data, page='dashboard')
    else:
        return redirect(url_for('login'))

@app.route("/prediksi", methods=['GET', 'POST'])
def prediksi():
    if 'username' in session:
        if request.method == 'POST':
            # Membaca data kecepatan angin dari file CSV
            data = pd.read_csv('kec_angin_reg.csv')

            # Preprocess the data
            # Mengubah data menjadi tipe numpy array
            data_angin = data['wind_speed'].values.reshape(-1, 1)

            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data_angin)

            # Define parameter
            look_back = int(request.form["look_back"])
            batch_value = int(request.form["batch"])
            epoch_value = int(request.form["epoch"])
            neuron_value = int(request.form["neuron"])
            training_value = float(request.form["training_size"])

            # Split the data into train and test sets
            train_size = int(len(scaled_data) * training_value)
            test_size = len(scaled_data) - train_size
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]

            # Prepare the train and test datasets
            def create_dataset(data):
                X, y = [], []
                for i in range(len(data) - look_back):
                    X.append(data[i:i+look_back])
                    y.append(data[i+look_back])
                return np.array(X), np.array(y)

            X_train, y_train = create_dataset(train_data)
            X_test, y_test = create_dataset(test_data)

            # Reshape the input data for LSTM
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            # Build the LSTM model
            model = Sequential()
            model.add(LSTM(units=neuron_value, return_sequences=True, input_shape=(look_back, 1)))
            model.add(LSTM(units=neuron_value))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            # Train the model
            model.fit(X_train, y_train, epochs=epoch_value, batch_size=batch_value)

            # define y_pred and y_test with true value
            y_pred = model.predict(X_test)
            y_pred = scaler.inverse_transform(y_pred)
            y_test = scaler.inverse_transform(y_test)
            
            # define y_predTrain and y_train with true value
            y_predTrain = model.predict(X_train)
            y_predTrain = scaler.inverse_transform(y_predTrain)
            y_train = scaler.inverse_transform(y_train)

            # Plot hasil prediksi
            trainPredictPlot = np.empty_like(data_angin)
            trainPredictPlot[:, :] = np.nan
            trainPredictPlot[look_back:len(y_predTrain) + look_back, :] = y_predTrain
            testPredictPlot = np.empty_like(data_angin)
            testPredictPlot[:, :] = np.nan
            testPredictPlot[len(y_predTrain) + (look_back * 2):len(data_angin), :] = y_pred

            plt.figure(figsize=(25, 10))
            plt.plot(data_angin, label='Data Asli')
            plt.plot(trainPredictPlot, label='Prediksi Pelatihan')
            plt.plot(testPredictPlot, label='Prediksi Pengujian')
            plt.legend()

            # Menghitung RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))

            # Menghitung MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_test - y_pred) / y_pred)) * 100

            # Menampilkan hasil evaluasi
            print("RMSE:", rmse)
            print("MAPE:", mape)

            # convert to dataframe
            df = pd.DataFrame(y_pred)
            df['Nilai_Aktual'] = pd.DataFrame(y_test)

            # Simpan plot ke dalam objek BytesIO
            img = BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plt.close()

            # Ubah objek BytesIO ke dalam format base64
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Prediksi 7 hari kedepan
            # Memanggil Data terakhir pada window size
            last_sequence = X_test[-1]

            # Membuat array untuk menyimpan data prediksi
            future_predictions = []
            time_steps = look_back
            n_features = 1

            # Mulai prediksi 3 hari
            for _ in range(24):  # 8*3 hari
                predicted_value = model.predict(
                    last_sequence.reshape(1, time_steps, n_features))
                future_predictions.append(predicted_value[0, 0])
                last_sequence = np.append(last_sequence[1:], predicted_value[0])

            # Mengembalikan data menjadi skala semula
            future_predictions = np.array(future_predictions).reshape(-1, 1)
            future_predictions = scaler.inverse_transform(future_predictions)

            # Menampilkan hasil prediksi
            last_date = data['datetime'].values[-1]
            waktu = []
            # Mengubah tanggal terakhir menjadi objek datetime
            last_date = pd.to_datetime(last_date, format='%d/%m/%Y %H:%M' )

            # Menambahkan 1 hari ke tanggal terakhir
            next_day = last_date + timedelta(hours=3)

            # Generate tanggal
            time_range = pd.date_range(
                next_day, periods=len(future_predictions), freq='3H')
            for i in range(len(future_predictions)):
                waktu.append(time_range[i].strftime("%Y-%m-%d %H:%M:%S"))

            # Menampilkan data ke web
            normalisasi = np.column_stack((data_angin.tolist(), scaled_data.tolist()))

            prediksi = np.column_stack((y_test.tolist(), y_pred.tolist()))
            # Menggabungkan tanggal dengan data prediksi menggunakan np.column_stack

            pred = np.column_stack((waktu, future_predictions))
            pred.tolist()

            # Print the model summary
            model.summary()
            
            a = model.layers[2].trainable_weights
            a_list = [w.numpy().tolist() for w in a[0]]

            model.save('lstm_model.h5')

            return render_template("prediksi.html", 
                                   app_data=app_data, 
                                   norm=normalisasi, 
                                   pk=prediksi, 
                                   pred=pred, 
                                   a=a_list, 
                                   rmse="{:.{}f}".format(rmse, 3), 
                                   mape="{:.{}f}".format(mape, 3), 
                                   plot_url=plot_url, 
                                   page='prediksi')
        
        return render_template("prediksi.html", 
                               app_data=app_data, 
                               form_input = True, 
                               page='prediksi')
    else:
        return redirect(url_for('login'))
  
@app.route("/download/<filename>")
def download(filename):
    return send_file(filename, as_attachment=True)

@app.route("/hasil", methods=['GET', 'POST'])
def hasil():
    if 'username' in session:
        if request.method == 'POST':
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                uploaded_file.save(uploaded_file.filename)

                input_df = pd.read_csv(uploaded_file.filename)

                # Load the trained model from model.h5
                # model = load_model('lstm_model.h5')
                model = load_model('lstm_optimal.h5')

                # Perform data preprocessing on the uploaded data
                data_angin = input_df['wind_speed'].values.reshape(-1, 1)

                # scaled_data_input = scaler.transform(data_angin)
                scaler = MinMaxScaler()
                scaled_data_input = scaler.fit_transform(data_angin)
                look_back = 8
                jml_pred = 8
                # jml_pred = float(request.form["jml_pred"])
                
                # Perform predictions for the next 10 days, with predictions made every 3 hours
                future_predictions = []
                time_steps = look_back
                n_features = 1
                last_sequence = scaled_data_input[-look_back:]
                
                for _ in range(jml_pred):  # 8 * 3 days
                    predicted_value = model.predict(
                        last_sequence.reshape(1, time_steps, n_features))
                    future_predictions.append(predicted_value[0, 0])
                    last_sequence = np.append(
                        last_sequence[1:], predicted_value[0])

                
                future_predictions = np.array(future_predictions).reshape(-1, 1)
                future_predictions = scaler.inverse_transform(future_predictions)

                # Generate datetime range for the predictions
                last_date = input_df['datetime'].values[-1]
                waktu = []

                # Mengubah tanggal terakhir menjadi objek datetime
                last_date = pd.to_datetime(last_date, format='%d/%m/%Y %H:%M' )

                # Menambahkan 1 hari ke tanggal terakhir
                next_day = last_date + timedelta(hours=3)

                time_range = pd.date_range(
                    next_day, periods=len(future_predictions), freq='3H')
                for i in range(len(future_predictions)):
                    waktu.append(time_range[i].strftime("%d-%m-%Y %H:%M"))

                # Menggabungkan tanggal dengan data prediksi menggunakan np.column_stack
                pred = np.column_stack((waktu, future_predictions))
                pred.tolist()

                normalisasi = np.column_stack(
                    (data_angin.tolist(), scaled_data_input.tolist()))
                
                csv_folder = 'download'
                xlsx_folder = 'download'

                # Simpan hasil prediksi sebagai file CSV
                csv_filename2 = f'{csv_folder}/prediksi_custom.csv'
                pd.DataFrame(pred, columns=['waktu', 'future_predictions']).to_csv(
                    csv_filename2, index=False)

                # Simpan hasil prediksi sebagai file XLSX
                xlsx_filename2 = f'{xlsx_folder}/prediksi_custom.xlsx'
                pd.DataFrame(pred, columns=['waktu', 'Normalisasi']).to_excel(
                    xlsx_filename2, index=False)

                # Simpan normalisasi sebagai file CSV
                csv_filename = f'{csv_folder}/normalisasi.csv'
                pd.DataFrame(normalisasi, columns=['Kecepatan Angin', 'Normalisasi']).to_csv(
                    csv_filename, index=False)

                # Simpan normalisasi sebagai file XLSX
                xlsx_filename = f'{xlsx_folder}/normalisasi.xlsx'
                pd.DataFrame(normalisasi, columns=['Kecepatan Angin', 'Normalisasi']).to_excel(
                    xlsx_filename, index=False)

                return render_template("hasil.html", 
                                       app_data=app_data, 
                                       args=True, 
                                       norm=normalisasi, 
                                       pred=pred, 
                                       csv_filename2=csv_filename2, 
                                       xlsx_filename2=xlsx_filename2, 
                                       csv_filename=csv_filename, 
                                       xlsx_filename=xlsx_filename, 
                                       page='hasil')

        return render_template("hasil.html", 
                               app_data=app_data, 
                               args=False, 
                               norm=[], 
                               pred=[], 
                               csv_filename2=None,
                               xlsx_filename2=None,
                               csv_filename=None, 
                               xlsx_filename=None, 
                               page='hasil')
    else:
        return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=DEVELOPMENT_ENV)
