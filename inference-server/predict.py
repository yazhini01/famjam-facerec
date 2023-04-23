from flask import Flask, jsonify, render_template, request
from fastbook import *
from fastai.vision.widgets import *
from contextlib import contextmanager
import pathlib

# uncomment in windows
# @contextmanager
# def set_posix_windows():
#     posix_backup = pathlib.PosixPath
#     try:
#         pathlib.PosixPath = pathlib.WindowsPath
#         yield
#     finally:
#         pathlib.PosixPath = posix_backup
# EXPORT_PATH = pathlib.Path("family.pkl")
# with set_posix_windows():
#     learn = load_learner(EXPORT_PATH)

learn = load_learner(pathlib.Path("family.pkl"))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = PILImage.create(img)
    img.show()
    pred, pred_idx, probs = learn.predict(img)
    return jsonify({
        'prediction': f'{pred}', 
        'probability': f'{probs[pred_idx]:.04f}'
    })

if __name__ == '__main__':
     app.run(port=80)