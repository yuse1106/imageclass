import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from keras.models import Sequential, load_model
import numpy as np
from PIL import Image
import sys

classes = ["monkey", "boar", "crow"]
num_class = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            #file.save(filepath)

            model = load_model("./animal_aug.cnn")

            image = Image.open(filepath)
            image = image.convert('RGB')
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X = []
            X.append(data)
            X = np.array(X)

            result = model.predict([X])[0]
            prediction = result.argmax()
            percent = int(result[prediction] * 100)
            label = "予測ラベル: " + classes[prediction]  
            output = "パーセント" + str(percent) + "%"
            return render_template('index.html', image=filepath,label=label,output=output)
            #return "予測ラベル: " + classes[prediction] + "パーセント" + str(percent) + "%"
            #return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!DOCTYPE html>
    <html lang=”ja”>
    <head>
    <meta charset=”UTF-8″>
    <title>画像分類アプリケーション</title>
    <style>
        body{
            font-family: Arial, sans-serif;
            text-align: center;
        }
        h1{
            color: #333;
        }
        #image-display{
            margin-top: 20px;
        }
        img{
            max-width: 100%;
        }
        #classification-result{
            margin-top: 20px;
            font-size: 18px;
            color: #0078d4;
        }
    </style>
    </head>
    <body>
    <h1>画像を判定しよう！</h1>
    <form method = post enctype = multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>

    <font color="red"><h2>アップロード画像</h2><font></span>
    <div class="image-display"> 
    <img src={{image}} alt="写真" width="200" height="200">
    </div>

    <font color="black"><h3>画像判定結果</h3></font>
    <div class="classification-result"></div>
    </div>
    </body>
    </html>

    '''

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
