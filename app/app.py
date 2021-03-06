import os

import cv2
import numpy as np
import torch
from flask import Flask, request, flash, redirect, url_for, render_template
from torch import nn
from werkzeug.utils import secure_filename

# from app import PneumoniaModel, preprocess

ROOT_FOLDER = os.path.dirname(__file__)

UPLOAD_FOLDER = os.path.join(ROOT_FOLDER, "static/uploads")
ML_CORE_FOLDER = os.path.join(os.path.dirname(ROOT_FOLDER), "ml_core")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = app.static_folder
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def preprocess(x, resize=128):
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    x = cv2.resize(x, (256, 256))
    x = cv2.GaussianBlur(x, (5, 5), 1, 1)
    x = cv2.resize(x, (resize, resize))
    x = x - (cv2.GaussianBlur(x, (7, 7), 1, 1) * 0.8)
    x = (x - x.mean()).astype(np.float64)
    x = (x / x.max()).astype(np.float64)
    x = x.reshape(resize, resize, 1)
    x = np.moveaxis(x, -1, 0).reshape((1,1,128,128))
    return x


class PneumoniaModel(nn.Module):
    def __init__(self, input_shape, dropout=0.1, n_filters=16):
        super(PneumoniaModel, self).__init__()
        self.conv1 = nn.Conv2d(1, n_filters, 3, 1)
        self.pool1 = nn.MaxPool2d((3, 3), stride=1)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(n_filters, n_filters // 2, (3, 3), 1)
        self.pool2 = nn.MaxPool2d((3, 3), stride=1)
        self.dropout2 = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_filters // 2 * (input_shape[0] - 8) * (input_shape[1] - 8), 32 * 32)
        self.activation1 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(32 * 32, 32 * 32)
        self.activation2 = nn.LeakyReLU()
        self.dropout4 = nn.Dropout(dropout)
        self.linear3 = nn.Linear(32 * 32, 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x, out=0):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        if out == -2:
            return x
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        if out == -1:
            return x
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout3(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout4(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return x


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return '''
    <!doctype html>
    <title>Upload a new chest X-Ray File</title>
    <h1>Upload a new chest X-Ray File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/results', methods=['GET'])
def uploaded_file():
    global model
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], request.args.get('filename'))
    img = preprocess(cv2.imread(img_path))
    print(img, img.shape)
    if torch.cuda.is_available():
        img = torch.tensor(img).cuda().float()
    else:
        img = torch.tensor(img).float()
    with torch.no_grad():
        pred = model(img)[0]
    print(pred, pred.shape)
    labels = ("Normal", "Pneumonia")
    idx = int(pred.argmax())
    user = dict(class_name=f"{labels[idx]}".lower(),
                confidence=f"{float(pred.max()):>5.2%}",
                img_path="static/"+request.args.get('filename'))
    return render_template("results.html", user=user)


if __name__ == '__main__':
    model = PneumoniaModel((128, 128), 0.4, 16)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(ML_CORE_FOLDER, "pneumodia_model.pt")))
    else:
        model.load_state_dict(torch.load(os.path.join(ML_CORE_FOLDER, "pneumodia_model.pt"),
                                         map_location=torch.device('cpu')))
    model.eval()
    app.run(debug=True, host="0.0.0.0")
