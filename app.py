from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory,send_file,redirect
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

def extract_feature(file, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file, res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result
def extract_feature_readfile(file, mfcc, chroma, mel):
    X, sample_rate = librosa.load(file, res_type='kaiser_fast')
    if chroma:
        stft=np.abs(librosa.stft(X))
    result=np.array([])
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
    if chroma:
        chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result
emotions={
  '01':'neutral',
  '02':'neutral',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
# Emotions to observe
observed_emotions=['neutral','calm','happy','sad','angry','fearful','disgust','surprised']
model_save_name = '84.25ACC.pkl'
model = pickle.load(open(model_save_name,'rb'))

app = Flask(__name__,static_url_path="", static_folder="static")
app.config['CORS_HEADERS'] = 'Content-Type'
CORS(app)

@app.route("/", methods=["GET",'POST'])
def make_predictions():
    transcript="not detected"
    if request.method == 'POST':
        print(request.files)
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            a=[]
            # feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
            print("done")
            a.append(feature)
            transcript=model.predict(np.array(a))

    return render_template('index.html',transcript=transcript)

if __name__ == "__main__":
    app.run()