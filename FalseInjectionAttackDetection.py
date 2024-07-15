from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from lsanomaly import LSAnomaly
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from filterpy import kalman

main = tkinter.Tk()
main.title("Prediction")
main.geometry("1000x650")

global filename, dataset, X, scaler

def uploadDataset():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Loaded\n\n')
    dataset = pd.read_csv(filename,nrows=1000)
    text.insert(END,str(dataset))

def preprocessDataset():
    text.delete('1.0', END)
    global X, dataset, scaler
    dataset.fillna(0, inplace = True)
    dataset.drop(['marker'], axis = 1,inplace=True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    dataset = dataset[indices_to_keep].astype(np.float64)
    dataset = dataset.sort_values(by=['R1-PA1:VH', 'R1-PM1:V'])
    dataset = dataset.values
    X = dataset[:,0:dataset.shape[1]-1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    text.insert(END,"Normalized Dataset values\n\n")
    text.insert(END, str(X))

def runLFA():
    global X,predict
    text.delete('1.0', END)
    lsanomaly = LSAnomaly(sigma=1, rho=0.01)
    lsanomaly.fit(X)
    predict = lsanomaly.predict(X)
    for i in range(len(predict)):
        if predict[i] == 'anomaly':
            predict[i] = 1
    for i in range(0,20):
        if predict[i] == 0:
            text.insert(END,str(X[i])+"=========>Normal\n\n")
        else:
            text.insert(END,str(X[i])+"=========>LFA Attack\n\n")
def LFA_graph():
    global X,predict
    text.update_idletasks()        
    plt.figure(figsize=(12,8))
    no_attack = [True if i == 0 else False for i in predict]
    attack = [True if i == 1 else False for i in predict]
    plt.title("LFA Normal & Attack Detection Graph")
    a = plt.scatter(X[no_attack, 0], X[no_attack, 1], c = 'black', edgecolor = 'k', s = 30, label='No Attack')
    b = plt.scatter(X[attack, 0], X[attack, 1], c = 'red', edgecolor = 'k', s = 30, label='LFA Attack')
    plt.plot(X[:,0], X[:,1], label='Attack Traces')
    plt.axis('tight')
    plt.xlabel('Timestep');
    plt.ylabel('Normal & Injected Values');
    plt.legend()
    plt.show()

def runNIA():
    global X,NIA_predict
    NIA_predict=[]
    text.delete('1.0', END)
    x, P = kalman.predict(x=X, P=X[:,0])
    NIA_predict = np.where(P > 0.5, 1, 0)
    for i in range(len(NIA_predict)):
        if predict[i] == 'anomaly':
            NIA_predict[i] = 1
    for i in range(0,20):
        if NIA_predict[i] == 0:
            text.insert(END,str(X[i])+"=========>Normal\n\n")
        else:
            text.insert(END,str(X[i])+"=========>NIA Attack\n\n")
    
def NIA_graph():
    global X,NIA_predict
    plt.figure(figsize=(12,8))
    no_attack = [True if i == 0 else False for i in NIA_predict]
    attack = [True if i == 1 else False for i in NIA_predict]
    plt.title("NIA Normal & Attack Detection Graph")
    a = plt.scatter(X[no_attack, 0], X[no_attack, 1], c = 'black', edgecolor = 'k', s = 30, label='No Attack')
    b = plt.scatter(X[attack, 0], X[attack, 1], c = 'red', edgecolor = 'k', s = 30, label='NIA Attack')
    plt.plot(X[:,0], X[:,1], label='Attack Traces')
    plt.axis('tight')
    plt.xlabel('Timestep');
    plt.ylabel('Normal & Injected Values');
    plt.legend()
    plt.show()
    
def Run_Predict():
    global filename, dataset
    filename = filedialog.askopenfilename(initialdir = "Dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' Test Data Loaded\n\n')
    dataset = pd.read_csv(filename)
    dataset.fillna(0, inplace = True)
    indices_to_keep = ~dataset.isin([np.nan, np.inf, -np.inf]).any(1)
    dataset = dataset[indices_to_keep].astype(np.float64)
    dataset = dataset.values
    X = dataset
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    lsanomaly = LSAnomaly(sigma=1, rho=0.01)
    lsanomaly.fit(X)
    LFA_predict = lsanomaly.predict(X)
    x, P = kalman.predict(x=X, P=X[:, 0])
    NIA_predict = np.where(P > 0.5, 1, 0)
    print(len(X))
    print(LFA_predict)
    print(NIA_predict)
    for i in range(len(LFA_predict)):        
        if LFA_predict[i] == 0 and NIA_predict[i] == 0:
            text.insert(END, str(X[i]) + "=========>Normal\n\n")
        elif LFA_predict[i] == 1 and NIA_predict[i] == 1:
            text.insert(END, str(X[i]) + "=========>Both LFA and NIA Attacks\n\n")
        elif LFA_predict[i] == 1 and NIA_predict[i] == 0:
            text.insert(END, str(X[i]) + "=========>LFA Attack\n\n")
        elif NIA_predict[i] == 1 and LFA_predict[i] == 0:
            text.insert(END, str(X[i]) + "=========>NIA Attack\n\n")


font = ('times', 15, 'bold')
title = Label(main, text='Safeguarding cyber-physical system:Detecting controlled LFA-NIA attacks', justify=LEFT)
#title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload False Injection Dataset", command=uploadDataset)
uploadButton.place(x=10,y=100)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Preprocess Dataset", command=preprocessDataset)
preprocessButton.place(x=300,y=100)
preprocessButton.config(font=font1)

lfaButton = Button(main, text="Run LFA Attack Detection", command=runLFA)
lfaButton.place(x=480,y=100)
lfaButton.config(font=font1)

lfaButton = Button(main, text="LFA Attack Detection Graph", command=LFA_graph)
lfaButton.place(x=710,y=100)
lfaButton.config(font=font1)

niaButton = Button(main, text="Run NIA Attack Detection", command=runNIA)
niaButton.place(x=10,y=150)
niaButton.config(font=font1)

niaButton = Button(main, text="NIA Attack Detection Graph", command=NIA_graph)
niaButton.place(x=300,y=150)
niaButton.config(font=font1)

exitButton = Button(main, text="Prediction", command=Run_Predict)
exitButton.place(x=710,y=150)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=160)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1) 

#main.config(bg='light coral')
main.mainloop()
