<b>Hey hi techie! WELCOME TO MY BREAST_CANCER_DETECTION PROJECT :)<br>
oh  wait!! yeah nothing. Haha you can go through this "README" and can read all my highlighted comments for that you can easily understand my code.<br>
Can't wait to see? AHAHA just kidding !! All the best buddy, Have a great day.</b><br>


<b># Breast_Cancer_Detection</b><br>

<b>#import libraries</b><br>
import numpy as np<br>
import sklearn.datasets<br>


<b>#Getting the dataset</b><br>
breast_cancer = sklearn.datasets.load_breast_cancer()<br>

print(breast_cancer)<br>

X = breast_cancer.data<br>
Y = breast_cancer.target<br>

print(X)<br>
print(Y)<br>

print(X.shape, Y.shape)<br>

import pandas as pd<br>

data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)<br>

data['class'] = breast_cancer.target<br>

data.head()<br>

data.describe()<br>

print(data['class'].value_counts())<br>

print(breast_cancer.target_names)<br>

data.groupby('class').mean()<br>


<b>"""0 - Malignant<br></b>

<b>1 - Benign<br></b>

<b>Now train and test split"""</b><br>



from sklearn.model_selection import train_test_split<br>

X_train, X_test, Y_train, Y_test = train_test_split(X,Y)<br>

print(Y.shape, Y_train.shape, Y_test.shape)<br>

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1)<br>
<b>#test_size-->to specify the percentage of test data needed</b><br>

print(Y.shape, Y_train.shape, Y_test.shape)<br>

print(Y.mean(), Y_train.mean(), Y_test.mean())<br>

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1, stratify=Y)<br>
<b>#stratify --> for correct distribution of the data as of the original data</b><br>

print(Y.mean(), Y_train.mean(), Y_test.mean())<br>

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1, stratify=Y, random_state=1)<br>
<b>#random_state --> specific split of data. each value of random state splits the data differently</b><br>

print(X_train.mean(), X_test.mean(), X.mean())<br>

print(X_train)<br>


<b>"""Logistic Regression"""<br>

#import Logistic Regression from sklearn</b><br>
from sklearn.linear_model import LogisticRegression<br>

classifier=LogisticRegression() #loading the logistic regression model to the variable "classifier"<br>

classifier.fit?<br>

<b>#training the model on training data</b><br>
classifier.fit(X_train, Y_train)<br>

<b>"""Evaluation of the model"""</b><br>


<b>#import accuracy_score</b><br>
from sklearn.metrics import accuracy_score<br>

prediction_on_training_data = classifier.predict(X_train)<br>
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)<br>

print("accuracy on training data: ",accuracy_on_training_data)<br>

<b>#prediction on test_data</b><br>
prediction_on_test_data=classifier.predict(X_test)<br>
accuracy_on_test_data=accuracy_score(Y_test, prediction_on_test_data)<br>

print("accuracy_on_test_data:", accuracy_on_test_data)<br>

<b>"""Detecting whether the patient has breast cancer in benign or Malignant"""</b><br>
input_data = (9.504,12.44,60.34,273.9,0.1024,0.06492,0.02956,0.02076,0.1815,0.06905,0.2773,0.9768,1.909,15.7,0.009606,0.01432,0.01985,0.01421,0.02027,0.002968,10.23,15.66,65.13,314.9,0.1324,0.1148,0.08867,0.06227,0.245,0.07773)<br>
<b>#change the input data to numpy array to make prediction</b><br>
input_data_as_numpy_array = np.asarray(input_data)<br>
print(input_data) <br>
<b>#reshape the array as we are predicting the output for one instance</b><br>
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)<br>

<b>#prediction</b><br>
prediction = classifier.predict(input_data_reshaped)<br>
print(prediction)  <b>#returns a list with element [0] if malignant; returns a list with [1], if benign.</b><br>

if (prediction[0]==0):<br>
  print("The breast cancer is malignant")<br>

else:<br>
  print("The breast cancer is benign")<br>
  
  
<b>"""YEAH CONGRATULATIONS YOU'VE DONE THIS!  :)"""</b>



