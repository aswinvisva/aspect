import io
import os
from math import pi

import flask
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category20c
from bokeh.plotting import figure
from bokeh.transform import cumsum
from flask import jsonify, request, Response, render_template, redirect
from tensorflow.python.keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils, inception_v3
from PIL import Image
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import uuid

import pandas as pd

from os.path import dirname, join

from bokeh.io import curdoc
from bokeh.embed import components
from bokeh.models.widgets import Tabs

app = flask.Flask(__name__)
app.config["DEBUG"] = True

feature_names = ["Negative Result",
                 "Mild DR",
                 "Moderate DR",
                 "Severe DR",
                 "Proliferate DR"
                 ]

database = {}
last_id=None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/dashboard/')
def recent_dashboard():
    return redirect("/dashboard/"+last_id)

@app.route('/dashboard/<page_id>')
def show_dashboard(page_id):

    if page_id not in database.keys():
        return "Patient not found!"

    x = {
        'No DR':database[page_id]["Prediction"][0],
        'Mild DR':database[page_id]["Prediction"][1],
        'Moderate DR': database[page_id]["Prediction"][2],
        'Severe DR': database[page_id]["Prediction"][3],
        'Proliferate DR': database[page_id]["Prediction"][4],
    }

    data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    data['color'] = Category20c[len(x)]

    p = figure(plot_height=350, title="AI Prediction", toolbar_location=None,
               tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='country', source=data)

    imgArray = database[page_id]["Image"].convert('RGBA')
    xdim, ydim = imgArray.size

    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    # Copy the RGBA image into view, flipping it so it comes right-side up
    # with a lower-left origin
    view[:, :, :] = np.flipud(np.asarray(imgArray))

    dim = max(xdim, ydim)
    image = figure(title="Uploaded Image",x_range=(0,dim), y_range=(0,dim))
    image.image_rgba(image=[img],x=0, y=0, dw=xdim, dh=ydim)

    script, div = components(p)
    image_script, image_div = components(image)

    return render_template('dashboard.html', script=script,
                           div=div, script2=image_script, div2=image_div, patient_id=page_id)


@app.route('/api/v1/predictor/diagnosis', methods=['POST'])
def api_diagnosis():
    # initialize the data dictionary that will be returned from the
    # view
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    data = {"success": False}
    model = load_model('detector_model.h5')
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":

        if flask.request.files.get("image"):
            patient_id = str(uuid.uuid1())
            database[patient_id] = {}

            global last_id
            last_id = patient_id

            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = image_preprocessing(image, patient_id)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            prediction = model.predict(image)

            database[patient_id]["Prediction"] = prediction[0]
            print(database[patient_id])

            # loop over the results and add them to the list of
            # returned predictions
            index = np.where(prediction == prediction.max())[1][0]
            r = {"label": float(index), "probability_distribution": prediction[0].tolist()}
            data["prediction"] = r

            # indicate that the request was a success
            data["success"] = True

        else:
            print("No image!")

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


def image_preprocessing(image, patient_id):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize((224, 224))

    database[patient_id]["Image"] = image

    image = img_to_array(image)

    image = inception_v3.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image

app.run()
