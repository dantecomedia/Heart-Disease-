import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dataset=pd.read_csv("heart.csv")

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X=sc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=None)






import keras
from keras.models import Sequential
from keras.layers import Dense


classifier=Sequential()
classifier.add(Dense(600, input_dim=13, kernel_initializer="uniform", activation="relu" ))
classifier.add(Dense(100,kernel_initializer="uniform", activation="relu" ))
classifier.add(Dense(1,kernel_initializer="uniform", activation="sigmoid"))
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
classifier.summary()


classifier.fit(X_train,Y_train, epochs=100, batch_size=5)

Y_pred=classifier.predict(np.array(X_test))

Y_pred=(Y_pred>0.5)

from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_pred,Y_test)

from sklearn.metrics import confusion_matrix
zz=confusion_matrix(Y_test,Y_pred)



    
