from flask import Flask, request, render_template, send_file
from flask_celery import make_celery
import time
from read_db import get_result, check_status
from style_transfer.model import style_transfer_image
import io 
import os
import tensorflow as tf
import numpy
import numpy as np
import base64
from celery.result import AsyncResult
from PIL import Image
from io import BytesIO




app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = broker='amqp://localhost//'
app.config['CELERY_RESULT_BACKEND'] =  'db+sqlite:///db.sqlite3'

celery = make_celery(app)

def load_img(img, max_dim):
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img.numpy()
    img = img[tf.newaxis, :]
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_photos():
    if request.method == 'POST':
        # retrieve the uploaded photos from the request
        photo1 = request.files['content Photo']
        photo2 = request.files['Style Photo']

        max_dim=1000
        photo1 = base64.b64encode(photo1.read()).decode('utf-8')
        photo2 = base64.b64encode(photo2.read()).decode('utf-8')

        res = transfer.delay(photo1, photo2)

        return {"id":res.id, "status": res.status}


    return render_template(r'upload.html', css_file='style.css')


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

@app.route('/download', methods=['GET', 'POST'])
def download_result():
    if request.method == 'POST':
        task_id = request.form['task_id']
        image = get_result(task_id)
        img_data = BytesIO(image)
        return send_file(img_data, mimetype='image/png', as_attachment=True)

    return render_template('download.html')

    # return send_file(img_io, mimetype='image/jpeg', as_attachment=True, attachment_filename='image.jpg')
       
    
    

@app.route('/status', methods=['GET'])
def get_task_status():
    if 'task_id' in request.args:
        task_id = request.args['task_id']
        task_result = AsyncResult(task_id, app=celery)

        return task_result.state

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
     app.run(port=8001)
