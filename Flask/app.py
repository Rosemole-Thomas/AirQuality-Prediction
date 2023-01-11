import pickle

import pandas as pd
from flask import Flask, jsonify, request, render_template


app = Flask(__name__)

with open('../air quality prediction/air quality prediction/forecast_model.pkl', 'rb') as fin:
    m2 = pickle.load(fin)

@app.route('/')  # Homepage
def home():
    return render_template('index.html')


@app.route('/Submit', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    # retrieving values from form
    horizon = int(request.form['horizon'])

    future2 = m2.make_future_dataframe(periods=horizon)
    forecast2 = m2.predict(future2)

    data = forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']][-horizon:]
    data_df=pd.DataFrame(data)
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    fig1 = m2.plot(forecast2)  # datenow = datetime.now()
    datenow = datetime(2020, 6, 2)
    dateend = datenow + timedelta(days=horizon)
    datestart = datetime(2018, 6, 2)
    plt.xlim([datestart, dateend])
    plt.title("AQI forecast", fontsize=20)
    plt.xlabel("Day", fontsize=20)
    plt.ylabel("AQI", fontsize=20)
    plt.axvline(datenow, color="k", linestyle=":")
    plt.show()



    return render_template('index.html',
                           prediction_text=plt.show()) # rendering the predicted result



if __name__ == "__main__":
    app.run(debug=True)


