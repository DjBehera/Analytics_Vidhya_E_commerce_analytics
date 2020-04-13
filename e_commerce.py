import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

###Loading all the classification guys
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb

### Loading Deep Learning brothers if needed
''''from keras.layers import Activation,BatchNormalization,Dropout,Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.callbacks import EarlyStopping'''
from sklearn.metrics import accuracy_score

import category_encoders as ce
from sklearn.model_selection import StratifiedKFold


print('[INFO]....Loading train and test data')
train_data = pd.read_csv('data/train_8wry4cB.csv')
test_data = pd.read_csv('data/test_Yix80N0.csv')
print('[INFO]....Completed')
merge_data = pd.concat([train_data,test_data])
merge_data = merge_data[train_data.columns]

def category_bin(data,num):
    category_list = set()
    for i in data.values:
        for j in i.split(';'):
            category_list.add(j.split('/')[num])
    return(category_list)
    
category_set = category_bin(merge_data.ProductList,0)
sub_category_set = category_bin(merge_data.ProductList,1)
sub_sub_category_set = category_bin(merge_data.ProductList,2)

def total_category_browsed(data):
    count_dic = {}
    for i in category_set:
        count_dic[i] = 0
    for i in data.split(';'):
        count_dic[i.split('/')[0]] += 1
    return(pd.Series(count_dic))

def total_sub_category_browsed(data):
    count_dic = {}
    for i in sub_category_set:
        count_dic[i] = 0
    for i in data.split(';'):
        count_dic[i.split('/')[1]] += 1
    return(pd.Series(count_dic))

def total_sub_sub_category_browsed(data):
    count_dic = {}
    for i in sub_sub_category_set:
        count_dic[i] = 0
    for i in data.split(';'):
        count_dic[i.split('/')[2]] += 1
    return(pd.Series(count_dic))

def category_encoding(encoding_columns,s,X,y,testset):
    print('[INFO]....Category Encoding in progress')
    oof = pd.DataFrame([])
    for tr_idx,val_idx in StratifiedKFold(n_splits=5,random_state=123,shuffle=True).split(X,y):
        ce_target_encoder = ce.TargetEncoder(encoding_columns,smoothing=s)
        ce_target_encoder.fit(X.iloc[tr_idx],y.iloc[tr_idx])
        oof = oof.append(ce_target_encoder.transform(X.iloc[val_idx]))
    ce_target_encoder = ce.TargetEncoder(encoding_columns,smoothing=s)
    ce_target_encoder.fit(X,y)
    testset = ce_target_encoder.transform(testset)
    X = oof.sort_index()
    print('[INFO]....Completed')
    return(X,testset)

def stacking_models(X_train,X_test,y_train,y_test):    
    
    base_model1 = lgb.LGBMClassifier()
    base_model1.fit(X_train,y_train)
    pred1 = base_model1.predict_proba(X_test)[:,1]
    
    base_model2_params = {'colsample_bytree': 1,
                          'learning_rate': 0.1,
                          'n_estimators': 50,
                          'num_leaves': 100,
                          'reg_alpha': 1,
                          'reg_lambda': 5,
                          'subsample': 0.7}
    base_model2 = lgb.LGBMClassifier(**base_model2_params)
    base_model2.fit(X_train,y_train)
    pred2 = base_model2.predict_proba(X_test)[:,1]
    
    base_model3_params = {'colsample_bytree': 0.7,
                          'learning_rate': 0.1,
                          'n_estimators': 50,
                          'num_leaves': 100,
                          'reg_alpha': 1,
                          'reg_lambda': 1,
                          'subsample': 0.7}
    base_model3= lgb.LGBMClassifier(**base_model3_params)
    base_model3.fit(X_train,y_train)
    pred3 = base_model3.predict_proba(X_test)[:,1]
    
    pred = np.column_stack((pred1,pred2,pred3))
    
    meta_model = LogisticRegression()
    meta_model.fit(pred,y_test)
    
    print(meta_model.coef_)
    return(base_model1,base_model2,base_model3,meta_model)
    

def cross_validation_lgboost(X_train,y_train):   
    grid_params = {
        'reg_lambda':[1,5,10],
        'reg_alpha':[1,5,10],
        'num_leaves':[50,100,150,200],
        'learning_rate':[0.3,0.1,0.05],
        'colsample_bytree':[0.7,0.9,1],
        'n_estimators': [25,50,100,200],
        'subsample':[0.7,0.9,0.1]
        }
    model = lgb.LGBMClassifier()
    cv = GridSearchCV(model, grid_params,
                        verbose=1,
                        cv=4,
                       n_jobs=-1,scoring='f1')
    cv.fit(X_train,y_train)
    print('lgb model score on training set : ',cv.best_score_)
    best_model_params.append(cv.best_params_)
    return(cv.best_estimator_)


def data_preprocessing(scaling=False):
    print('[INFO]....Data preprocessing in progress')
    merge_data = pd.concat([train_data,test_data],ignore_index=True)
    merge_data = merge_data[train_data.columns]
    merge_data.startTime = pd.to_datetime(merge_data.startTime)
    merge_data.endTime = pd.to_datetime(merge_data.endTime)
    
    print('[INFO]....Addding new features')
    merge_data['no_of_products_viewed'] = merge_data.ProductList.apply(lambda x:len(x.split(';')))
    merge_data['session_duration'] = (merge_data.endTime - merge_data.startTime).apply(lambda x:x.seconds // 60)
    
    for i in list(category_set):
        merge_data[i] = 0
    merge_data.iloc[:,7:] = merge_data.ProductList.apply(total_category_browsed)
    test = merge_data.iloc[:,7:].sum(axis=1)
    merge_data.iloc[:,7:] = merge_data.iloc[:,7:].div(test,axis=0)
    
    merge_data['hour'] = merge_data.startTime.dt.hour
    merge_data['weekday'] = merge_data.startTime.dt.weekday
    merge_data['month'] = merge_data.startTime.dt.month
    
    for i in sub_category_set:
        merge_data[i] = 0
    merge_data.iloc[:,-86:] = merge_data.ProductList.apply(total_sub_category_browsed)
    
    for i in sub_sub_category_set:
        merge_data[i] = 0
    merge_data.iloc[:,-383:] = merge_data.ProductList.apply(total_sub_sub_category_browsed)
    
    merge_data['avg_browse_time_on_product'] = merge_data.session_duration / merge_data.no_of_products_viewed
    
    merge_data = pd.get_dummies(columns=['month','weekday','hour'],data=merge_data,drop_first=True)
    
    merge_data['first_item_browsed'] = merge_data.ProductList.apply(lambda x : x.split(';')[0].split('/')[-2])
    
    merge_data.drop(columns=['startTime','endTime','session_id','ProductList'],inplace=True)
    merge_data.gender = merge_data.gender.map({'female':1,'male':0})
    
    if scaling:
        print('Enabling Standard scaling ......')
        cols_to_be_scaled = [i for i in merge_data.columns if i.startswith('B') | i.startswith('C')]
        cols_to_be_scaled.extend(['no_of_products_viewed','session_duration','avg_browse_time_on_product'])
        sc = StandardScaler()
        merge_data[cols_to_be_scaled] = sc.fit_transform(merge_data[cols_to_be_scaled])
    print('[INFO]....Data preprocessing is completed')
    
    return(merge_data[:len(train_data)],merge_data[len(train_data):].drop(columns=['gender']))

def data_prediction():
    train,test = data_preprocessing()
    X = train.drop(columns=['gender'])
    y = train['gender']
    print('[INFO]....trainset shape: ',X.shape)
    print('[INFO]....testset shape: ',test.shape)
    encoding_columns = ['first_item_browsed']

    X,test = category_encoding(encoding_columns,0.2,X,y,test)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)
    
    ##########################FOR BASE LGBM############################################
    '''model = lgb.LGBMClassifier()
    model.fit(X_train,y_train)
    print('score on validation data: ',model.score(X_test,y_test))
    final_pred = model.predict(test)'''
    
    ##########################FOR LGBM USING RFECV#########################################
    print('[INFO]....Creating an LGBM model')
    print('[INFO]....Applying RFECV to select 150 features')
    
    model = lgb.LGBMClassifier()
    model = RFECV(estimator=model,step=10,min_features_to_select=150,scoring='accuracy')
    model.fit(X_train,y_train)
    
    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    test = model.transform(test)
    
    print('[INFO]....After tranformation train shape :',X_train.shape)
    
    model = lgb.LGBMClassifier()
    model.fit(X_train,y_train)
    print('score on validation data: ',model.score(X_test,y_test))
    final_pred = model.predict(test)
    
    ###########################FOR STACKING PURPOSE ############################################

    '''basemodel_1,basemodel_2,basemodel_3,meta_model = stacking_models(X_train,X_test,y_train,y_test)
    base_pred_test = np.column_stack((basemodel_1.predict_proba(test)[:,1],basemodel_2.predict_proba(test)[:,1],\
                                     basemodel_3.predict_proba(test)[:,1]))
    
    final_pred = meta_model.predict(base_pred_test)'''
    
    ###########################FOR NEURAL NETWORK PURPOSE#############################################
    #model = neural_net(X_train,y_train,X_train.shape[1])
    
    
    #pd.Series(dict(zip(X.columns.tolist(),model.feature_importances_))).sort_values(ascending=False).head(20).plot(kind='bar')
    return(final_pred)

def create_csv(y_pred):
    print('[INFO]....Storing the prediction on submission.csv in same folder')
    prediction = pd.DataFrame({'session_id':test_data.session_id,'gender':y_pred})
    prediction.gender = prediction.gender.map({1:'female',0:'male'})
    prediction.to_csv('submission.csv',index=False)
    print('[EXIT]')
    
if __name__ == '__main__':
    y_pred = data_prediction()
    create_csv(y_pred)
    

