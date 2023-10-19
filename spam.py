import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
#Loading data
data = pd.read_csv("C:\\Users\\HP\\Desktop\\DATA SCIENCE\\Email Spam\\spam.csv",encoding='ISO-8859-1')
# print(data)
#preprocessing
mail_data = data.where((pd.notnull(data)),'')
# print(mail_data.head())
# print(mail_data.shape)
#label encoding
mail_data.loc[mail_data['v1']=='spam','v1'] = 0
mail_data.loc[mail_data['v1']=='ham','v1'] = 1
# print(mail_data)
#seperating the datas based on category and message
x = mail_data['v2']
y = mail_data['v1']
# print(x)
# print(y)
# train and test spliting of datas
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)
# print(x_train.shape)
#print(x_test.shape)
#feature extraction(transforming text data into numerical data)
fe_ext = TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
x_train_fe = fe_ext.fit_transform(x_train)
x_test_fe = fe_ext.transform(x_test)
# ytrain and ytest values as integer
y_train = y_train.astype('int')
y_test = y_test.astype('int')
# print(x_train_fe)
#logistic regression
model = LogisticRegression()
# training the regression model
model.fit(x_train_fe,y_train)
# evaluating the trained model and making prediction
pre_data = model.predict(x_train_fe)
accuracy_data = accuracy_score(y_train,pre_data)
print('Accuracy on training data :',accuracy_data)
# evaluating the test model and making prediction
pre_data = model.predict(x_test_fe)
accuracy_data = accuracy_score(y_test,pre_data)
print('Accuracy on test data :',accuracy_data)
# building predictive system
input_mail = ["Sunshine Quiz Wkly Q! Win a top Sony DVD player if u know which country the Algarve is in? Txt ansr to 82277. å£1.50 SP:Tyrone,,,"]
# convert text to feature vector
inp_fe_data = fe_ext.transform(input_mail)
# predecting 
prediction = model.predict(inp_fe_data)
print(prediction)
if prediction[0]==1:
    print('Ham mail')
else:
    print('Spam mail')
