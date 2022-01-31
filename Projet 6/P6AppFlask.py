# *****************************************************
#
# Project : App Flask - Reconnaissance d'image Qualité
# Béton
# Project 6 - POC
# Auteur : Ali Naama
#
# *****************************************************



import os
import uuid
import flask
import urllib
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from keras.preprocessing import image
from flask import Flask, render_template, request, send_file
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import codecs, json


from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import io

global model
global pathmdl
global graph
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pathmdl  =  str(BASE_DIR) + '\p6.model'
print(BASE_DIR)
print(pathmdl)
model = load_model(pathmdl)

#model = load_model(BASE_DIR + '/model/p6.h5')

ALLOWED_EXT = set(['jpg', 'jpeg', 'png', 'jfif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


classes = ['shade','efflorescence','ressuage','Black','pommeling','Rust','dustiness','stone','milt','bubbling','cracking','flaking']


def load():
    # load the pre-trained Keras model
    model = load_model(pathmdl)

    graph = tf.get_default_graph()


def prepare_image(image, target):
    # resize the input image and preprocess it
    image = cv2.resize(np.array(image), target)
    #image = np.array(image).reshape(None, 64, 64, 3)
    #image = image.astype('float32')
    #image = image / 255.

    return image

def predict(filename, model):

    test_image = image.load_img(filename, target_size=(64, 64))
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    print(result)
    dict_result = {}
    for i in range(11):
        dict_result[result[0][i]] = classes[i]
        print(classes[i])
        #print(result[i][i])

    res = result[0]
    print(res)
    #res.sort()
    #print(res)
    #res = res[::-1]
    #print(res)
    prob = res[:11]
    print(prob)

    prob_result = []
    class_result = []
    for i in range(11):
        prob_result.append((prob[i] * 100).round(2))
        print((prob[i] * 100).round(2))
        class_result.append(classes[i])
        #print(dict_result[(prob[i] * 100).round(2)])
        print((prob[i] * 100).round(2))
        print(classes[i])
        #class_result.append(dict_result[(prob[i] * 100).round(2)])

    return class_result, prob_result


@app.route('/')
def home():
    return render_template("index2.html")



@app.route('/success', methods=['GET', 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd(), 'static/images')
    if request.method == 'POST':
        if (request.form):
            link = request.form.get('link')
            try:
                resource = urllib.request.urlopen(link)
                unique_filename = str(uuid.uuid4())
                filename = unique_filename + ".jpg"
                img_path = os.path.join(target_img, filename)
                output = open(img_path, "wb")
                output.write(resource.read())
                output.close()
                img = filename

                class_result, prob_result = predict(img_path, model)
                print(class_result)
                print(prob_result)
                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "class4": class_result[3],
                    "class5": class_result[4],
                    "class6": class_result[5],
                    "class7": class_result[6],
                    "class8": class_result[7],
                    "class9": class_result[8],
                    "class10": class_result[9],
                    "class11": class_result[10],
                    "class12": class_result[11],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                    "prob4": prob_result[3],
                    "prob5": prob_result[4],
                    "prob6": prob_result[5],
                    "prob7": prob_result[6],
                    "prob8": prob_result[7],
                    "prob9": prob_result[8],
                    "prob10": prob_result[9],
                    "prob11": prob_result[10],
                    "prob12": prob_result[11]
                }

            except Exception as e:
                print(str(e))
                error = 'This image from this site is not accesible or inappropriate input'

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index2.html', error=error)


        elif (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img, file.filename))
                img_path = os.path.join(target_img, file.filename)
                img = file.filename

                class_result, prob_result = predict(img_path, model)

                predictions = {
                    "class1": class_result[0],
                    "class2": class_result[1],
                    "class3": class_result[2],
                    "class4": class_result[3],
                    "class5": class_result[4],
                    "class6": class_result[5],
                    "class7": class_result[6],
                    "class8": class_result[7],
                    "class9": class_result[8],
                    "class10": class_result[9],
                    "class11": class_result[10],
                    "prob1": prob_result[0],
                    "prob2": prob_result[1],
                    "prob3": prob_result[2],
                    "prob4": prob_result[3],
                    "prob5": prob_result[4],
                    "prob6": prob_result[5],
                    "prob7": prob_result[6],
                    "prob8": prob_result[7],
                    "prob9": prob_result[8],
                    "prob10": prob_result[9],
                    "prob11": prob_result[10]

                }

            else:
                error = "Please upload images of jpg , jpeg and png extension only"

            if (len(error) == 0):
                return render_template('success.html', img=img, predictions=predictions)
            else:
                return render_template('index2.html', error=error)

    else:
        return render_template('index2.html')


@app.route("/predictapi", methods=["POST"])
def predictapi():
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            img = flask.request.files["image"].read()
            img = Image.open(io.BytesIO(img))
            #image = prepare_image(image, target=(64, 64))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((64, 64))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            inputs = preprocess_input(img)
            #predictions = model.predict(inputs)
            predictions = model.predict(inputs)
            print(predictions)
            b = predictions.tolist()  # nested lists with same data, indices
            json_predictions = json.dumps(b)
    # return the results as a JSON response
    return flask.jsonify(json_predictions)



if __name__ == "__main__":
    app.run(debug=True)


