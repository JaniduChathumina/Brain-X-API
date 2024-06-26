from flask import Flask, request, jsonify, send_file
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from flask_cors import CORS
import LimeExplainer 
import GradCAMExplainer
import traceback
from flask_swagger_ui import get_swaggerui_blueprint
from urllib.request import urlretrieve

app = Flask(__name__)
CORS(app) 

#python 3.10.2
#print("TensorFlow version:", tf.__version__) #2.16.1
#print("NumPy version:", np.__version__) #1.23.5


SWAGGER_URL = '/api/docs'  # URL for Swagger UI
API_URL = '/static/docs.json'  # URL for API documentation

# Configuring Swagger UI to point API documentation
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "BrainX API"
    }
)

# Registering the Swagger UI blueprint with the flask app
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


# Define the URL of the model file in Firebase Storage
model_url = 'https://firebasestorage.googleapis.com/v0/b/api-model-2f5ae.appspot.com/o/model_2.h5?alt=media&token=https://firebasestorage.googleapis.com/v0/b/api-model-2f5ae.appspot.com/o/model_2.h5?alt=media&token=c1883887-ea06-4373-a10e-a539f1cb82ac'

local_model_path, _ = urlretrieve(model_url, "model_2.h5")

# Load the model from the local file path
model = tf.keras.models.load_model(local_model_path)


# Loading the pre-trained model
# model = tf.keras.models.load_model('model_2.h5')
class_list = ['Astrocitoma','Carcinoma','Ependimoma','Ganglioglioma','Germinoma','Glioblastoma','Granuloma','Meduloblastoma','Meningioma','Neurocitoma','Oligodendroglioma','Papiloma','Schwannoma','Tuberculoma','_NORMAL']
most_likely_class : int

# image preprocessing function
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(600, 600))
    img = np.expand_dims(img, axis=0)
    return img

# predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        file = request.files['file']
        img_path = 'inputImage.jpeg'
        file.save(img_path)

        # img_path = 'astrocitoma.jpeg'
        processed_img = preprocess_image(img_path)

        # Make a prediction using the loaded model
        prediction = model.predict(processed_img)

        global most_likely_class 
        most_likely_class = int(np.argmax(prediction))
        result_label = class_list[most_likely_class]
        result_value = prediction[0][most_likely_class]
        result_value = round(result_value * 100, 2)

        # Return the prediction as JSON
        return jsonify({'prediction': result_label,'confedience':str(result_value)+'%'})

    except Exception as e:
        return jsonify({'error': str(e)})

# lime explanation text endpoint
@app.route('/limeExplanationText', methods=['GET'])
def limeExplanationText():
    try:
        lime_explainer = LimeExplainer.LimeExplainer()
        img_path = 'inputImage.jpeg' 
        fig, output_data = lime_explainer.explain_fn(img_path, model)
        return jsonify({'explanation':output_data})

    except Exception as e:
        return jsonify({'error': str(e)})

# lime explanation image endpoint
@app.route('/limeExplanationImage', methods=['GET'])
def limeExplanationImage():
    try:
        lime_image_path = 'output_lime_image.jpeg'
        image_response = send_file(lime_image_path, mimetype='image/jpeg')
        return image_response

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/gradcamExplanation', methods=['GET'])
def gradcamExplanation():
    try:
        img_path = 'inputImage.jpeg'
        gradcam_explainer = GradCAMExplainer.GradCAMExplainer()
        heatmap, score = gradcam_explainer.explain_fn(img_path, most_likely_class, model, class_list)
        heatmap_image_path = 'heatmap.jpeg'
        heatmap_response = send_file(heatmap_image_path, mimetype='image/jpeg')
        return heatmap_response
    
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)})
    
@app.route('/gradcamExplanationMask', methods=['GET'])
def gradcamExplanationMask():
    try:
        score_image_path = 'score.jpeg'
        mask_response = send_file(score_image_path, mimetype='image/jpeg')
        return mask_response

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)})



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
