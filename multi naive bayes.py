# -*- coding: utf-8 -*-
"""
Created on Fri May 28 14:08:16 2021

@author: ilike
"""
import data_preparation as dp
import transformation as trans
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
#from keras.callbacks import Callback


num_of_classes = 18

corpous = dp.get_prepared_data(num_of_classes, None, 200,100, True, True, True)

x = corpous.clean_text
y = corpous.book_id


#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

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
                
       avg_acc_score = sum(acc_score)/10
       
       print('accuracy of each fold - {}'.format(acc_score))
       print('Avg accuracy : {}'.format(avg_acc_score))
            
       '''
 
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
        '''
        
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
                
       avg_acc_score = sum(acc_score)/10
       
       print('accuracy of each fold - {}'.format(acc_score))
       print('Avg accuracy : {}'.format(avg_acc_score))
       '''
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
        '''
BOW_model(x,y)
tf_idf_model(x,y)        
        
#---------------------------------------------------------------------------       
'''        
def W2vec_model(x,y):
        word2vec = trans.Word2Vec(x)
        word2vec.build_vocab(x)
        word2vec.train(x, total_examples=len(x), epochs=1000)
        
        x_train,x_test,y_train,y_test=train_test_split(word2vec.wv.syn0,y,test_size=0.2,random_state=40)
        
        model = MultinomialNB()
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        
        print('accuracy %s' % accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
  '''      
   
#W2vec_model(x, y)
       
#-----------------------------------------------------------------------
