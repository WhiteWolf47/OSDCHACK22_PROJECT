from flask import Flask, render_template, request
import pickle
import numpy as np
from predictor import predict_with_model()
from names import get_names()


app = Flask(__name__)

@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    #data2 = request.form['b']
    #data3 = request.form['c']
    #data4 = request.form['d']
    #arr = np.array([[data1, data2, data3, data4]])
    model = tf.keras.models.load_model('./Models_simp2')
    img_path = str(data1)
    prediction = predict_with_model(model, img_path)
    path_to_names = r"./test_images"
    names = get_names(path_to_names)
    print(f"Character's Name : {names[prediction]}
    
	
    #pred = model.predict(arr)
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)