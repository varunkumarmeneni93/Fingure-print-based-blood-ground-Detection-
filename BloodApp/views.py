from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import io
import base64
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import os

global cnn_model, labels, accuracy, precision, recall, fscore

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 32):
        params = {'ksize':(ksize, ksize), 'sigma':1.0, 'theta':theta, 'lambd':15.0,
                  'gamma':0.02, 'psi':0, 'ktype':cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern,params))
    return filters

def getGabor(img, filters):
    results = []
    for kern,params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        results.append(fimg)
    return np.asarray(results[31])

path = "Dataset"
labels = []
X = []
Y = []

for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

filters = build_filters()

#function to calculate all metrics
def calculateMetrics(algorithm, y_test, predict):
    global accuracy, precision, recall, fscore
    accuracy = accuracy_score(y_test,predict)*100
    precision = precision_score(y_test, predict,average='macro') * 100
    recall = recall_score(y_test, predict,average='macro') * 100
    fscore = f1_score(y_test, predict,average='macro') * 100
    
X = np.load('model/X.npy')
Y = np.load('model/Y.npy')

X = X.astype('float32')
X = X/255

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test

cnn_model = Sequential()
cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(units = 256, activation = 'relu'))
cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
if os.path.exists("model/cnn_weights.hdf5") == False:
    model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
    hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 40, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
    f = open('model/cnn_history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()    
else:
    cnn_model.load_weights("model/cnn_weights.hdf5")

predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
calculateMetrics("CNN", y_test1, predict)

def TrainML(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore
        output='<table border=1 align=center width=100%><tr><th><font size="" color="black">Algorithm Name</th><th><font size="" color="black">Accuracy</th>'
        output += '<th><font size="" color="black">Precision</th><th><font size="" color="black">Recall</th><th><font size="" color="black">FSCORE</th>'
        output+='</tr>'
        algorithms = ['Deep Learning CNN Algorithm']
        output += '<td><font size="" color="black">'+algorithms[0]+'</td><td><font size="" color="black">'+str(accuracy)+'</td><td><font size="" color="black">'+str(precision)+'</td>'
        output += '<td><font size="" color="black">'+str(recall)+'</td><td><font size="" color="black">'+str(fscore)+'</td></tr>'
        output+= "</table></br>"
        f = open('model/cnn_history.pckl', 'rb')
        train_data = pickle.load(f)
        f.close()
        plt.figure(figsize=(6,3))
        plt.grid(True)
        plt.xlabel('EPOCH')
        plt.ylabel('Accuracy')
        plt.plot(train_data['accuracy'], color = 'green')
        plt.plot(train_data['val_accuracy'], color = 'blue')
        plt.plot(train_data['loss'], color = 'black')
        plt.plot(train_data['val_loss'], color = 'red')
        plt.legend(['Train Accuracy', 'Validation Accuracy','Train Loss','Validation Loss'], loc='upper left')
        plt.title('Deep Learning CNN Training Accuracy & Loss Graph')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        #plt.close()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def PredictAction(request):
    if request.method == 'POST':
        global labels, filters
        cnn_model = load_model("model/cnn_weights.hdf5")
        filename = request.FILES['t1'].name
        image = request.FILES['t1'].read() #reading uploaded file from user
        if os.path.exists("BloodApp/static/"+filename):
            os.remove("BloodApp/static/"+filename)
        with open("BloodApp/static/"+filename, "wb") as file:
            file.write(image)
        file.close()
        img = cv2.imread("BloodApp/static/"+filename)
        img = getGabor(img, filters)
        img = cv2.resize(img, (32, 32))
        img = img.reshape(1,32,32,3)#convert image as 4 dimension
        img = img.astype('float32')#convert image features as float
        img = img/255 #normalized image
        predict = cnn_model.predict(img)#now predict dog breed
        predict = np.argmax(predict)
        img = cv2.imread("BloodApp/static/"+filename)
        img = cv2.resize(img, (500,450))#display image with predicted output
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 0, 255), 2)
        output = '<font size="4" color="blue">Predicted As : '+labels[predict]+"</font>"
        plt.figure(figsize=(6, 4))
        plt.imshow(img)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        plt.clf()
        plt.cla()
        #plt.close()
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def Predict(request):
    if request.method == 'GET':
       return render(request, 'Predict.html', {})    

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})    

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})   

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        if username == 'admin' and password == 'admin':
            context= {'data':'welcome '+username}
            return render(request, 'UserScreen.html', context)
        else:
            context= {'data':'login failed'}
            return render(request, 'UserLogin.html', context)        
    
