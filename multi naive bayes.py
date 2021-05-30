# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:08:16 2021

@author: ilike
"""
import data_preparation as dp
import transformation as trans
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report,accuracy_score,plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt  
import numpy as np


num_of_classes = 5

corpous = dp.get_prepared_data(num_of_classes, None, 200,100, True, True, True)

x = corpous.clean_text
y = corpous.book_id




def BOW_model(x,y):

       acc_score = []
       kf = KFold(n_splits=10, shuffle= True ,random_state=42)
       for train_index, test_index in kf.split(x):
               X_train, X_test = x[train_index], x[test_index]
               y_train, y_test = y[train_index], y[test_index]
               BOW = trans.bag_of_words(X_train)
               x_train_trans = BOW.transform(X_train)
               x_test_trans = BOW.transform(X_test)
                
               Encoder = LabelEncoder()
               y_train = Encoder.fit_transform(y_train)
               y_test = Encoder.fit_transform(y_test)
                
               model = MultinomialNB()
               model.fit(x_train_trans,y_train)
               y_pred = model.predict(x_test_trans)
               acc = accuracy_score(y_pred , y_test)
               acc_score.append(acc)
       acc ={}         
       avg_acc_score = sum(acc_score)/10
       acc["BOW_after_cv"] = avg_acc_score
       
       print('accuracy of each fold - {}'.format(acc_score))
       print('Avg accuracy : {}'.format(avg_acc_score))
       
       plot_confusion_matrix(model,x_test_trans, y_test)  
       plt.show()
            
      #-------------------------- Before CV ---------------------------------------------------------- 
       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
        
       BOW = trans.bag_of_words(x_train)
       x_train_trans = BOW.transform(x_train)
       x_test_trans = BOW.transform(x_test)
        
       Encoder = LabelEncoder()
       y_train = Encoder.fit_transform(y_train)
       y_test = Encoder.fit_transform(y_test)
        
       model = MultinomialNB()
       model.fit(x_train_trans,y_train)
       y_pred = model.predict(x_test_trans)
        
       print('accuracy %s' % accuracy_score(y_test, y_pred))
       print(classification_report(y_test, y_pred))
       acc['BOW_before_cv'] = accuracy_score(y_test, y_pred)
       
       plot_confusion_matrix(model,x_test_trans, y_test)  
       plt.show()
       return acc
def tf_idf_model(x,y):
       acc_score = []
       kf = KFold(n_splits=10, shuffle= True ,random_state=42)
       for train_index, test_index in kf.split(x):
               X_train, X_test = x[train_index], x[test_index]
               y_train, y_test = y[train_index], y[test_index]
               tf_idf = trans.tf_idf(X_train)
               x_train_trans = tf_idf.transform(X_train)
               x_test_trans = tf_idf.transform(X_test)
                
               Encoder = LabelEncoder()
               y_train = Encoder.fit_transform(y_train)
               y_test = Encoder.fit_transform(y_test)
                
               model = MultinomialNB()
               model.fit(x_train_trans,y_train)
               y_pred = model.predict(x_test_trans)
               acc = accuracy_score(y_pred , y_test)
               acc_score.append(acc)
       
       acc = {}
       avg_acc_score = sum(acc_score)/10
       acc['TF_IDF_after_cv'] = avg_acc_score

       print('accuracy of each fold - {}'.format(acc_score))
       print('Avg accuracy : {}'.format(avg_acc_score))
       
       plot_confusion_matrix(model,x_test_trans, y_test)  
       plt.show()
       #------------------------------Before CV --------------------------------------------
       x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)
       
       tf_idf = trans.tf_idf(x_train)
       x_train_trans = tf_idf.transform(x_train)
       x_test_trans = tf_idf.transform(x_test)
        
       Encoder = LabelEncoder()
       y_train = Encoder.fit_transform(y_train)
       y_test = Encoder.fit_transform(y_test)
        
       model = MultinomialNB()
       model.fit(x_train_trans,y_train)
        
       y_pred = model.predict(x_test_trans)
        
       print('accuracy %s' % accuracy_score(y_test, y_pred))
       print(classification_report(y_test, y_pred))
       acc['TF_IDF_before_cv'] = accuracy_score(y_test, y_pred)
       
       plot_confusion_matrix(model,x_test_trans, y_test)  
       plt.show()
       return acc 
acc = BOW_model(x,y)
acc_t = tf_idf_model(x,y)        

acc.update(acc_t)
myList = acc.items()
myList = sorted(myList)
x_axis, y_axis = zip(*myList)

plt.plot(x_axis, y_axis)
plt.xlabel('Transformation models')
plt.ylabel('Accuracy')
plt.title('Comparison between transformation models with CV & without CV')
plt.show()
    