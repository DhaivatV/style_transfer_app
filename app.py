from flask import Flask, request, render_template, send_file
from flask_celery import make_celery
from read_db import get_result
from model import style_transfer_image
import tensorflow as tf
import numpy
import numpy as np
import base64
from celery.result import AsyncResult
from PIL import Image
from io import BytesIO


app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = broker='redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] =  'db+sqlite:///db.sqlite3'

celery = make_celery(app)

@app.route('/', methods=['GET'])
def index():
    
    return render_template(r'index.html', css_file='style.css')

@app.route('/upload', methods=['GET', 'POST'])
def upload_photos():
    if request.method == 'POST':
        # retrieve the uploaded photos from the request
        photo1 = request.files['content Photo']
        photo2 = request.files['Style Photo']
     
        photo1 = base64.b64encode(photo1.read()).decode('utf-8')
        photo2 = base64.b64encode(photo2.read()).decode('utf-8')

        res = transfer.delay(photo1, photo2)
    
        return render_template('up_response.html', task_id=res.id, status=res.status)


    return render_template(r'upload.html', css_file='style.css')


@app.route('/download', methods=['GET', 'POST'])
def download_result():
    if request.method == 'POST':
        task_id = request.form['task_id']
        image = get_result(task_id)
        img_data = BytesIO(image)
        return send_file(img_data, mimetype='image/png')

    return render_template('download.html')

   

@app.route('/status', methods=['GET'])
def get_task_status():
    if 'task_id' in request.args:
        task_id = request.args['task_id']
        task_result = AsyncResult(task_id, app=celery)

        return render_template('get_status.html', status=task_result.state, )

    return render_template('status.html')


@celery.task(name = 'app.transfer')
def transfer(photo1, photo2):
    print('request accepted')
    photo1 = base64.b64decode(photo1.encode('utf-8'))
    photo2 = base64.b64decode(photo2.encode('utf-8'))
    img = style_transfer_image(
            photo1, photo2, save_name="output_img",
            style_weight=1e-2, content_weight=3e4, total_variation_weight=30,
        )
    
    return img



if __name__=='__main__':
     app.run(debug=False)
