import streamlit as st
from sklearn.impute import SimpleImputer
from category_encoders import OneHotEncoder,OrdinalEncoder
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from preprocess import cardinal_info,drop
from sklearn.metrics import mean_absolute_error
import numpy as np

#@st.cache(suppress_st_warning=True)
def drop_col(df,high_columns):
    col_5,col_6,col_7 = st.columns([33,33,33])
    cardinal_col = cardinal_info(df)
    with col_5:
        cardinal = st.checkbox('Drop High Cardinal and Low Cardinal Features')
    with col_6:
        high_miss = st.checkbox('Drop Features High Percentage of Missing Values')
    with col_7:
        misc = st.checkbox('Drop Leaky, UnNecessary, MultiCollinear and Features Not Selected')
    if cardinal:
        with col_5:
            st.write(f'The columns to be dropped are :{[x for x in cardinal_col]},')
            drop_it_1 = st.radio('High Cardinality Value Feature',['Drop','Ignore'],horizontal=True)
            if drop_it_1 == 'Drop':
                df = drop(df,cardinal_col)
            else:
                pass
    if high_miss:
        with col_6:
            st.write(f'The columns to be dropped are :{[x for x in high_columns]},')
            drop_it_2 = st.radio('High Cardinality Value Feature',['Drop','Edit Features and Drop','Ignore'],horizontal=True,index=2)
            if drop_it_2 == 'Drop':
                df = drop(df,high_columns)
            elif drop_it_2 == 'Edit Features and Drop':
                feat = st.multiselect('Select Feature to Drop',options=high_columns)
                df = drop(df,feat)
    if misc:
        with col_7:
            st.write('Drop UnNeccesary and Leaky Features')
            avail_col = df.columns
            feat = st.multiselect('Select Feature to Drop',options=avail_col)
            df = drop(df,feat)
    
    return df
    
#@st.cache(suppress_st_warning=True)
def mean_fill(columns,X_train,X_test):
    mean_fill_feat = st.multiselect('Select Features to fill with Mean Values',options=columns)
    mean_impute = SimpleImputer(strategy='mean')
    if len(mean_fill_feat)<1:
        pass
    else:
        X_train[mean_fill_feat] = mean_impute.fit_transform(X_train[mean_fill_feat])
        X_test[mean_fill_feat] = mean_impute.transform(X_test[mean_fill_feat])
    return X_train,X_test

#@st.cache(suppress_st_warning=True)
def median_fill(columns,X_train,X_test):
    median_fill_feat = st.multiselect('Select Features to  fill with Median Values',options=columns)
    median_impute = SimpleImputer(strategy='median')
    if len(median_fill_feat)<1:
        pass
    else:
        X_train[median_fill_feat] = median_impute.fit_transform(X_train[median_fill_feat])
        X_test[median_fill_feat] = median_impute.transform(X_test[median_fill_feat])
    return X_train,X_test
#@st.cache(suppress_st_warning=True)
def mode_fill(columns,X_train,X_test):
    mode_fill_feat = st.multiselect('Select Features to fill with Most Frequent Values',options=columns)
    mode_impute = SimpleImputer(strategy='most_frequent')
    if len(mode_fill_feat)<1:
        pass
    else:
        X_train[mode_fill_feat] = mode_impute.fit_transform(X_train[mode_fill_feat])
        X_test[mode_fill_feat] = mode_impute.transform(X_test[mode_fill_feat])
    return X_train,X_test

#@st.cache(suppress_st_warning=True)
def ohe_transform(column,X_train,X_test):
    ohe_col = st.multiselect('Select Features to One Hot Encode',options=column)
    ohe = OneHotEncoder(use_cat_names=True,cols=ohe_col)
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)
    return X_train,X_test
#@st.cache(suppress_st_warning=True)
def ord_transform(column,X_train,X_test):
    ord_col = st.multiselect('Select Features to Ordinal Encode',options=column)
    ordl = OrdinalEncoder(cols=ord_col)
    X_train = ordl.fit_transform(X_train)
    X_test = ordl.transform(X_test)
    return X_train,X_test
#@st.cache(suppress_st_warning=True)
def label_transform(task,y_train,y_test):
    if task == 'Classification':
        target =LabelEncoder()
        y_train = target.fit_transform(y_train)
        y_test = target.transform(y_test)
    return y_train,y_test
#@st.cache(suppress_st_warning=True)
def split(column,df):
    target = st.selectbox('Select Your Target Variable',options=column)
    y = df[target]
    X = df.drop(target,axis=1)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    return X_train,X_test,y_train,y_test

def algorithms(task,selected_algorithm,regression_algorithm,Classification_algorithm):
    col_20,col_21,col_22,col_23,col_24= st.columns([5,35,15,35,10])
    
    
   
    
    
    
    if task == 'Regression':
        with col_21 :
            selected_algorithm = st.multiselect(f'Select the {task} Algorithm you want to use',options=regression_algorithm.keys())
            if len(selected_algorithm) <len(regression_algorithm.keys()):
                selected_algorithm.append(None)
        with col_23:
            length = len(selected_algorithm)
            st.write(f'The Following {length} Algorithm Selected Below Will be Used for The Modelling of the DataSet')
            for index in range(len(selected_algorithm)):
                st.info(f'{index+1} {selected_algorithm[index]}')
    else:
        with col_21 :
            selected_algorithm = st.multiselect(f'Select the {task} Algorithm you want to use',options=Classification_algorithm.keys())
            if len(selected_algorithm) <len(regression_algorithm.keys()):
                selected_algorithm.append(None)
        
        with col_23:
            length = len(selected_algorithm)
            st.write(f'The Following {length} Algorithm Selected Below Will be Used for The Modelling of the DataSet')
            for index in range(len(selected_algorithm)):
                st.info(f'{index+1} {selected_algorithm[index]}')
      
    return selected_algorithm
@st.cache(allow_output_mutation=True)
def model_data(task,model,model_evals,X_train,y_train):
    if task == 'Regression':
        if model_evals == 'Cross-Validation':
            model_val = cross_validate(model,X_train,y_train,cv=5,return_train_score=True,scoring=['r2','neg_mean_absolute_error'])
            train_scored = np.mean(model_val['train_r2'])
            val_scored = np.mean(model_val['test_r2'])
            train_score = np.mean((model_val['train_neg_mean_absolute_error'])*-1)
            val_score = np.mean((model_val['test_neg_mean_absolute_error'])*-1)
            return [train_score,val_score,train_scored,val_scored]
            #model_val.fit_transform
        elif model_evals == 'Train-Test-Split':
            X_trainer,X_val,y_trainer,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
            model.fit(X_trainer,y_trainer)
            model.fit(X_trainer,y_trainer)
            train_scored = model.score(X_trainer,y_trainer)
            val_scored = model.score(X_val,y_val)
            y_train_pred = model.predict(X_trainer)
            train_score = mean_absolute_error(y_trainer, y_train_pred)
            y_val_pred = model.predict(X_val)
            val_score = mean_absolute_error(y_val, y_val_pred)
            return [train_score,val_score,train_scored,val_scored]
        else:
            st.stop()
    else:
        if model_evals == 'Cross-Validation':
            model_val = cross_validate(model,X_train,y_train,cv=5,return_train_score=True,scoring='accuracy')
            train_score = np.mean(model_val['train_score'])
            val_score = np.mean(model_val['test_score'])
            return [train_score,val_score,None,None]
        elif model_evals == 'Train-Test-Split':
            X_trainer,X_val,y_trainer,y_val = train_test_split(X_train,y_train,test_size=0.2,random_state=42)
            model.fit(X_trainer,y_trainer)
            train_score = model.score(X_trainer,y_trainer)
            val_score = model.score(X_val,y_val)
            return [train_score,val_score,None,None]
        else:
            st.stop()
#@st.cache(allow_output_mutation=True)
def train_model(task,algo_dict,select,X_train,y_train,train_scores,selects,model_evals,model_eval,val_scores,train_scoreds,val_scoreds):
    dictionary = algo_dict[task]
    if model_eval=='Train-Test-Split':
        model = dictionary[select]
        score = model_data(task, model, model_eval, X_train, y_train)
        train_score,val_score = score[0],score[1]
        train_scored,val_scored = score[2],score[3]
        selects.append(select)
        train_scores.append(train_score)
        val_scores.append(val_score)
        model_evals.append(model_eval)
        train_scoreds.append(train_scored)
        val_scoreds.append(val_scored)
        return [train_scored,val_scored]
    elif model_eval == 'Cross-Validation':
        model = dictionary[select]
        score = model_data(task, model, model_eval, X_train, y_train)
        train_score,val_score = score[0],score[1]
        train_scored,val_scored = score[2],score[3]
        selects.append(select)
        train_scores.append(train_score)
        val_scores.append(val_score)
        model_evals.append(model_eval)
        train_scoreds.append(train_scored)
        val_scoreds.append(val_scored)
        return [train_scored,val_scored]
    else:
        return [None,None]

        
        