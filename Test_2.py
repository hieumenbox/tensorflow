import tensorflow as tf
from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request
from tensorflow import keras



# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'




@app.route('/predict', methods=['POST', 'GET'])
@cross_origin(origin='*')
def predict():
    model = keras.models.load_model('Model2.h5')
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    scaler = StandardScaler()
    value1 = float(request.args.get('value1'))
    value2 = float(request.args.get('value2'))
    value3 = float(request.args.get('value3'))
    value4 = float(request.args.get('value4'))
    value5 = float(request.args.get('value5'))
    value6 = float(request.args.get('value6'))
    value7 = float(request.args.get('value7'))
    value8 = float(request.args.get('value8'))
    value9 = float(request.args.get('value9'))
    value10 = float(request.args.get('value10'))

    test_value = np.array([[value1],
                           [value2],
                           [value3],
                           [value4],
                           [value5],
                           [value6],
                           [value7],
                           [value8],
                           [value9],
                           [value10]])
    test_value_1 = test_value.mean()
    # Test
    print(test_value)
    print(test_value_1)
    # Scale value
    scaler = scaler.fit(test_value)
    test_value = scaler.transform(test_value)
    test_value = np.array([test_value])
    test_value.shape
    pred_value = model.predict(test_value)
    pred_value = scaler.inverse_transform(pred_value)
    pred_value = pred_value.mean()
    mae = abs(test_value_1 - pred_value)
    print(test_value_1)      
    return str(test_value_1) + "a" + str(pred_value) + "b" + str(mae) + "c"


# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='6868')