import io
import os
import sqlite3
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
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from collections import Counter

import pandas as pd

from os.path import dirname, join

from bokeh.io import curdoc
from bokeh.embed import components
from bokeh.models.widgets import Tabs
from werkzeug.utils import secure_filename
from flask_migrate import Migrate

app = flask.Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:////tmp/patients.db'
db = SQLAlchemy(app)
migrate = Migrate(app, db)

last_id = None

class EyePatient(db.Model):
    __tablename__ = 'patients'
    id_key = db.Column(db.Integer, primary_key=True)
    date_added = db.Column(db.Date)
    id = db.Column(db.String(120))
    image_url = db.Column(db.String(120))
    no_dr_prob = db.Column(db.Float)
    mild_dr_prob = db.Column(db.Float)
    moderate_dr_prob = db.Column(db.Float)
    severe_dr_prob = db.Column(db.Float)
    proliferate_dr_prob = db.Column(db.Float)
    diagnosis = db.relationship("Diagnosis", backref="patients")

    def __repr__(self):
        return '<User %r>' % self.id

class Diagnosis(db.Model):
    __tablename__ = 'diagnosis'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id_key'))
    doctor_diagnosis = db.Column(db.String(120))
    doctor_comments = db.Column(db.String(120))
    date_added = db.Column(db.Date)

db.create_all()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/patient_landing', methods=['GET'])
def patient_landing():
    return render_template('patient_landing.html')

@app.route('/doctor_signup', methods=['GET'])
def doctor_signup():
    return render_template('doctor_signup.html')

@app.route('/dashboard/')
def recent_dashboard():
    return redirect("/dashboard/" + last_id)

@app.route('/patients/')
def get_patients():
    patients = EyePatient.query.all()

    return render_template("patients.html", patients=patients)

@app.route('/api/v1/add_diagnosis', methods=['POST'])
def add_diagnosis():
    id = search = request.args.get("id")
    diagnosis = request.form.get('diagnosis')
    comments = request.form.get('comments')

    db.session.add(Diagnosis(patient_id=id,
                             doctor_diagnosis=diagnosis,
                             doctor_comments=comments,
                             date_added=datetime.now()
                            ))
    db.session.commit()

    data = {}
    data["response"] = 200

    return flask.jsonify(data)

@app.route('/dashboard/<page_id>')
def show_dashboard(page_id):
    patient = EyePatient.query.filter_by(id_key=page_id).first()
    overall_diagnosis = Diagnosis.query.filter_by(patient_id=page_id)

    diagnosis = [d.doctor_diagnosis for d in overall_diagnosis]
    sentiment = Counter(diagnosis)
    number_of_diagnoses = len(diagnosis)

    if patient is None:
        return "Patient not found!"
    x = {
        'No DR': patient.no_dr_prob,
        'Mild DR': patient.mild_dr_prob,
        'Moderate DR': patient.moderate_dr_prob,
        'Severe DR': patient.severe_dr_prob,
        'Proliferate DR': patient.proliferate_dr_prob
    }

    data = pd.Series(x).reset_index(name='value').rename(columns={'index': 'country'})
    data['angle'] = data['value'] / data['value'].sum() * 2 * pi
    data['color'] = Category20c[len(x)]

    p = figure(plot_height=350, title="AI Prediction", toolbar_location=None,
               tools="hover", tooltips="@country: @value", x_range=(-0.5, 1.0))

    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
            line_color="white", fill_color='color', legend_field='country', source=data)

    im = Image.open(patient.image_url)
    imgArray = im.convert('RGBA')
    xdim, ydim = imgArray.size

    img = np.empty((ydim, xdim), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
    # Copy the RGBA image into view, flipping it so it comes right-side up
    # with a lower-left origin
    view[:, :, :] = np.flipud(np.asarray(imgArray))

    dim = max(xdim, ydim)
    image = figure(title="Uploaded Image", x_range=(0, dim), y_range=(0, dim))
    image.image_rgba(image=[img], x=0, y=0, dw=xdim, dh=ydim)

    script, div = components(p)
    image_script, image_div = components(image)

    return render_template('dashboard.html', script=script,
                           div=div, script2=image_script, div2=image_div, patient_id=page_id,
                           sentiment=sentiment, number=number_of_diagnoses, diagnosis=overall_diagnosis)


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

            global last_id
            last_id = patient_id

            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image.save(os.path.join('saved_images', secure_filename(patient_id) + '.jpg'), 'JPEG')

            # preprocess the image and prepare it for classification
            image = image_preprocessing(image, patient_id)

            # classify the input image and then initialize the list
            # of predictions to return to the client
            prediction = model.predict(image)

            db.session.add(EyePatient(id=patient_id,
                                      date_added=datetime.now(),
                                      image_url=str(os.path.join('saved_images', secure_filename(patient_id) + '.jpg')),
                                      no_dr_prob=float(prediction[0][0]),
                                      mild_dr_prob=float(prediction[0][1]),
                                      moderate_dr_prob=float(prediction[0][2]),
                                      severe_dr_prob=float(prediction[0][3]),
                                      proliferate_dr_prob=float(prediction[0][4])
                                      ))
            db.session.commit()

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

    image = img_to_array(image)

    image = inception_v3.preprocess_input(image)

    image = np.expand_dims(image, axis=0)

    return image


if __name__ == '__main__':
    app.run()
