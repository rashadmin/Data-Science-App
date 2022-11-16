import streamlit as st
import math
import pandas as pd
import numpy as np
from preprocess import  import_file,info,row_drop_na
from app_functions import mean_fill,median_fill,mode_fill,ohe_transform,ord_transform,label_transform,split,drop_col,algorithms,train_model
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression,LogisticRegression
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
import plotly.express as px
from plot_chart import plot_discrete_ordinal,feat_target_plot,target_plot,plot_continuous,nvn_relationship_plot,nvc_relationship_plot,cvc_relationship_plot
st.set_page_config(layout='wide')
cols = st.columns([20,60,20])
with cols[1]:
    st.title('DATA AGNOSTIC DATA SCIENCE APP')
st.subheader('Select DataSet')
select_file = st.radio('',options=[None,'Pick Available DataSet','Upload DataSet'],horizontal=True,label_visibility='collapsed')
if select_file == 'Upload DataSet':
    file = st.file_uploader('Upload Your File')
    if file is not None:
        file_ext = st.radio('Select File Extension',options = [None,'Csv','Xlsx'],horizontal=True)
        if file_ext is None:
            st.stop()
        elif file_ext=='Csv':
            delimiter = st.radio('Select Delimiter',options=[None,';',','],horizontal=True)
            if delimiter is None:
                st.stop()
            else:
                df = import_file(file,delimiter=delimiter)
        else:
            pass
    else:
        st.stop()
elif select_file == 'Pick Available DataSet':
    file = st.selectbox('Select DataSet', options = ['Crash Data','Real Estate Valuation','Placement Data','Student Score'] )
    if file =='Crash Data':
        df = pd.read_csv('data/Crash_Data.csv')
    elif file == 'Placement Data':
        df = pd.read_csv('data/Placement_Data_Full_Class.csv')
    elif file == 'Real Estate Valuation':
        df = pd.read_csv('data/Real estate valuation data set.csv')
    else:
        df = pd.read_csv('data/student-mat.csv',delimiter=';')
else:
    st.stop()
st.dataframe(df.head())

show_items = st.radio('',options=['Show DataFrame Information','Exploratory Data Analysis','Data Preprocessing and Modelling'],index=0,horizontal=True)
shape,describe,columns,high_columns = info(df)
if show_items=='Show DataFrame Information' :
    left_col,right_col = st.columns([50,50])
    col_1,col_2,col_3,col_4= st.columns([25,25,10,40])
    with left_col:
        st.subheader('DataFrame Information')
        st.write(f'The Shape of the DataFrame is {shape}')
        if len(columns) <=0:
            st.write('There are No Columns with Missing Values in this DataFrame')
        else:
            with left_col:
                st.write('The Columns of the DataFrame with Missing Values are:')
                st.caption('The Columns of the DataFrame with Red Background are very Important to Drop')
                length =len(columns)
                row = (math.ceil(length/2))

                for i in range(length):
                    if i<row:
                        with col_1:
                            if columns[i] in high_columns:
                                st.error(f'{columns[i]}')
                            else:
                                st.info(f'- {columns[i]}')
                    else:
                        with col_2:
                            if columns[i] in high_columns:
                                st.error(f'{columns[i]}')
                            else:
                                st.info(f'- {columns[i]}')
    with right_col:
        st.subheader('DataFrame Description')
        st.write('The Description of the DataFrame is:')
    with col_4:
        st.dataframe(describe)


if show_items == 'Exploratory Data Analysis':
    selecting,numerical,categorical,Target,nvn_relationship,nvc_relationship,cvc_relationship,feat_target =st.tabs(['Selecting Feature Types','Numerical Data','Categorical Data','Target Data','Numerical vs Numerical Relationship'
                                                                                                                    ,'Catergorical vs Numerical Relationship','Catergorical vs Catergorical Relationship','Features vs Target Relationship'])
    with selecting:
        col_50 = st.columns([2,30,3,30,3,30,2]) 
        with col_50[1]:
            all_col = df.columns
            st.write('Select Categorical Features')
            cat_col = st.multiselect('Select Categorical Features',options=all_col)
        with col_50[3]:
             st.write('Select Numerical Features')
             col_remain = df.drop(cat_col,axis=1).columns
             num_col = st.multiselect('Select Numerical Features',options=col_remain )
        with col_50[5]:
            #col_remain_2 = df.drop(cat_col + col_remain,axis=1)
            st.write('Select Target')

            col_remain_2 = df.drop(cat_col+num_col,axis=1).columns
            target = st.selectbox('Select Target Features',options=col_remain_2 )
        with col_50[6]:
            next_pg = st.checkbox('DONE')
        if not next_pg:
            st.stop()
    with numerical:
        #wer check for all numerical column and no object column
        col_51 = st.columns([2,30,3,30,3,30,2]) 
        with col_51[1]:
            st.write('Select Discrete Features')
            dis_col = st.multiselect('Select Discrete Features',options=num_col)
        with col_51[3]:
             st.write('Select Continuous Features')
             pre_cont_col = df[num_col].drop(dis_col,axis=1).columns
             cont_col = st.multiselect('Select Continuous Features',options=pre_cont_col)
        with col_51[5]:
            st.write('Select Feature to Ignore')

            pre_ignore_col = df[num_col].drop(cont_col+dis_col,axis=1).columns
            ignore = st.multiselect('Select Numerical Features to Ignore',options=pre_ignore_col )
        if len(dis_col)+len(cont_col)+len(ignore) == len(num_col):
            col_52 = st.columns([1,35,23,35,5]) 
            with col_52[1]:
                if len(dis_col) >0:
                    key = 'discrete'
                    st.write('Select Discrete Features to Plot')
                    dis_col_plot = st.selectbox('Select Discrete Features to Plot',options=dis_col)
                    fig = plot_discrete_ordinal(dis_col_plot, df,key)
                    with col_52[1]:
                        st.plotly_chart(fig)
            with col_52[3]:
                if len(cont_col) >0:
                     st.write('Select Continuous Features to Plot')
                     cont_col_plot = st.selectbox('Select Continuous Features to Plot',options=cont_col)
                     cont_fig = plot_continuous(cont_col_plot, df)
                     with col_52[3]:
                         st.plotly_chart(cont_fig)
    if len(cat_col) >0:
        with categorical:
            col_53 = st.columns([2,30,3,30,3,30,2]) 
            with col_53[1]:
                st.write('Select Nominal Features')
                nom_col = st.multiselect('Select Nominal Features',options=cat_col)
            with col_53[3]:
                 st.write('Select Ordinal Features')
                 pre_ord_col = df[cat_col].drop(nom_col,axis=1).columns
                 ord_col = st.multiselect('Select Ordinal Features',options=pre_ord_col)
            with col_53[5]:
                st.write('Select Feature to Ignore')
    
                pre_ignore_cat_col = df[cat_col].drop(nom_col+ord_col,axis=1).columns
                cat_ignore = st.multiselect('Select Categorical Features to Ignore',options=pre_ignore_cat_col )
            if len(nom_col)+len(ord_col)+len(cat_ignore) == len(cat_col):
                col_54 = st.columns([1,35,23,35,5]) 
                with col_54[1]:
                    if len(nom_col) >0:
                        key='nominal'
                        st.write('Select Nominal Features to Plot')
                        nom_col_plot = st.selectbox('Select Discrete Features to Plot',options=nom_col)
                        fig = plot_discrete_ordinal(nom_col_plot, df,key)
                        with col_54[1]:
                            st.plotly_chart(fig)
                with col_54[3]:
                    if len(ord_col) >0:
                         key='ordinal'
                         st.write('Select Ordinal Features to Plot')
                         ord_col_plot = st.selectbox('Select Continuous Features to Plot',options=ord_col)
                         ord_fig = plot_discrete_ordinal(ord_col_plot, df,key)
                         with col_54[3]:
                             st.plotly_chart(ord_fig)
    with Target:
        col_top = st.columns(2)
        with col_top[0]:
            task_plot = st.radio('What Machine Learning Task', options=['Regression','Classification'],horizontal=True)
        target_plot(task_plot, df, target)
    with nvn_relationship:
        col_56 = st.columns([5,10,30,30,20]) 
        with col_56[1]:
            dis_1 = st.checkbox('Discrete?',help='Check Discrete if First Feature is Discrete')
        with col_56[2]:
            num_col_1_plot = st.selectbox('Select First Numerical Features to Plot',options=num_col)
        with col_56[3]:
            num_col_remain = df[num_col].drop(num_col_1_plot,axis=1).columns
            num_col_2_plot = st.selectbox('Select Second Numerical Features to Plot',options=num_col_remain)
        with col_56[4]:
            dis_2 = st.checkbox('Discrete?',help='Check Discrete if Second Feature is Discrete')
        nvn_relationship_plot(df,dis_1,dis_2,num_col_1_plot,num_col_2_plot)
        
        ## q-q plot
        ## scatter plot
    if len(cat_col) >0 and len(num_col)>0:
        with nvc_relationship:
            col_57 = st.columns([5,10,30,30,20]) 
            with col_57[1]:
                dis_1 = st.checkbox('Discrete?',help='Check if Numerical Feature is Discrete')
            with col_57[2]:
                n_col_1_plot = st.selectbox('Select Numerical Features to Plot',options=num_col)
            with col_57[3]:
                c_col_1_plot = st.selectbox('Select Categorical Features to Plot',options=cat_col)
            #bar plot
            nvc_relationship_plot(df,dis_1,n_col_1_plot,c_col_1_plot)
            #box plot
    if len(cat_col) >0:
        with cvc_relationship:
            col_56 = st.columns([5,10,30,30,20]) 
            with col_56[2]:
                cat_col_1_plot = st.selectbox('Select First Numerical Features to Plot',options=cat_col)
            with col_56[3]:
                cat_col_remain = df[cat_col].drop(cat_col_1_plot,axis=1).columns
                cat_col_2_plot = st.selectbox('Select Second Numerical Features to Plot',options=cat_col_remain)
            cvc_relationship_plot(df, cat_col_1_plot, cat_col_2_plot)
    with feat_target:
        col_57 = st.columns([5,10,30,30,20]) 
        with col_57[2]:
            cat_col_tar_plot = st.selectbox('Select Categorical Feature to Plot Against Target',options=cat_col)
        with col_57[3]:
            num_col_tar_plot = st.selectbox('Select Numerical Feature to Plot Against Target',options=num_col)
        if len(num_col)==0:
            feat_target_plot(df, num_col_tar_plot, cat_col_tar_plot, task_plot, target,num_miss=True)
        elif len(cat_col)==0:
            feat_target_plot(df, num_col_tar_plot, cat_col_tar_plot, task_plot, target,cat_miss=True)
        else:
            feat_target_plot(df, num_col_tar_plot, cat_col_tar_plot, task_plot, target)
if show_items == 'Data Preprocessing and Modelling':
    dropping,splitting,filling,transforming,set_baseline,modelling =st.tabs(['Dropping','Splitting','Filling','Transforming','BaseLine Setting','Modelling'])
    with dropping:
       
        df = drop_col(df, high_columns)
        st.write(f'We have {len(df.columns)} Features Left!')
        st.dataframe(df)
    
    
    with splitting:
        col_12,col_13,col_14,col_15 = st.columns([10,40,40,10])
        column = df.columns
        with col_13:
            st.write('Split the Dataset into X and Y variable')
            X_train,X_test,y_train,y_test = split(column,df)
        with col_14:
            #st.components.v1.html("""<br>""")
            st.write('Select Your Machine Learning Task')
            task = st.selectbox('Select Your Machine Learning Task',options=['Regression','Classification'])
        with col_13:
            st.subheader('X_Train')
            st.dataframe(X_train.head())
        with col_14:
            st.subheader('X_Test')
            st.dataframe(X_test.head())
        if task == 'Regression':
            with col_13:
                st.subheader('Y_Train')
                st.dataframe(y_train.head())
            with col_14:
                st.subheader('Y_Test')
                st.dataframe(y_test.head())
        else:
            with col_13:
                st.subheader('Y_Train Value Count')
                st.dataframe(y_train.value_counts())
            with col_14:
                st.subheader('Y_Test Value Count')
                st.dataframe(y_test.value_counts())
        #st.dataframe(X)
    
    with filling:
        col_8,col_9,col_10,col_11 = st.columns(4)
        with col_8:
            st.write('Drop Rows With Missing Values')
            shape,describe,columns,high_columns = info(X_train)
            drop_row_feat = st.multiselect('Select Features to Drop Rows',options=columns)
            X_train = row_drop_na(X_train,drop_row_feat)
            X_test = row_drop_na(X_test,drop_row_feat)
            y_train = y_train.loc[X_train.index]
            y_test = y_test.loc[X_test.index]
            
        with col_9:
            shape,describe,columns,high_columns = info(X_train)
            st.write('Fill Rows With Missing Values with Mean')
            X_train,X_test = mean_fill(columns,X_train,X_test)
            
        with col_10:
            shape,describe,columns,high_columns = info(X_train)
            st.write('Fill Rows With Missing Values with Median')
            X_train,X_test = median_fill(columns, X_train, X_test)
            
        
        with col_11:
            shape,describe,columns,high_columns = info(X_train)
            st.write('Fill Rows With Missing Values with Most Frequent')
            X_train,X_test = mode_fill(columns, X_train, X_test)
            
        st.write(f'We have {len(X_train)} rows Left!')
        st.write(f'We have {len(y_train)} rows Left!')
        st.dataframe(X_train)
        if np.any(X_train.isnull()) and np.any(X_test.isnull()):
            st.stop()
        
    with transforming:
        col_16,col_17,col_18,col_19 = st.columns([10,40,40,10])
        column = X_train.columns
        with col_17:
            X_train,X_test = ohe_transform(column, X_train, X_test)
        with col_18:
            X_train,X_test = ord_transform(column, X_train, X_test)
        y_train,y_test = label_transform(task, y_train, y_test)
        st.dataframe(X_train,width=800)
        with col_19:
            next_page_1 = st.checkbox('Go to Baseline Page')
        if not next_page_1:
            st.stop()
        
    with set_baseline:
        col_61 = st.columns([80,20])
        selected_algorithm = []
        if task == 'Regression':
            baseline = [y_train.mean()]*len(y_train)
            baseline_score = mean_absolute_error(y_train,baseline)
        elif task == 'Classification':
            baseline_score = pd.Series(y_train).value_counts(normalize=True).max()
        with col_61[0]:
            st.subheader(f'The Baseline Score for this Data is {baseline_score}')
        with col_61[1]:
            next_page_2 = st.checkbox('Go to Training Model Page')
        regression_algorithm = {'Linear_Regression':LinearRegression(),'Decison_Tree_Regression':DecisionTreeRegressor(),
                                    'Random_Forest_Regression':RandomForestRegressor(),'Xgboost_Regression':XGBRFRegressor()}
        Classification_algorithm = {'Logistic Regression':LogisticRegression(),'Decision_Tree_Classifier':DecisionTreeClassifier(),
                                    'Random_Forest_Classifier':RandomForestClassifier(),'Xgboost_Classifier':XGBClassifier()}
        
        algo_dict = {'Regression':regression_algorithm,'Classification':Classification_algorithm}
        selected_algorithm = algorithms(task,selected_algorithm,regression_algorithm=regression_algorithm,Classification_algorithm=Classification_algorithm)
        st.dataframe(X_train)
        if not next_page_2:
            st.stop()
    
    with modelling:
        
        col_23 = st.columns(4)
        col_24 = st.columns(2)
        check = np.array([False]*len(selected_algorithm))
        train_scores,selects,model_evals,val_scores,train_scoreds,val_scoreds= [],[],[],[],[],[]
        with col_23[0]:
            select_0 = selected_algorithm[0]
            if select_0 is not None:
                check[0] = st.checkbox(f'{selected_algorithm[0]}',key=select_0)
                if check[0]:
                    model_eval = st.radio(f'Select Model Evaluation Type for {select_0}',['None','Cross-Validation','Train-Test-Split'],index=0,horizontal=True)
                    scores_0 =train_model( task, algo_dict, select_0, X_train, y_train,train_scores,selects,model_evals,model_eval,val_scores,train_scoreds,val_scoreds)
                    #train_score,val_score = scores_0[0],scores_0[1]
                
            else:
                st.write('No Algorithms Selected for this box')
        with col_23[1]:
            select_1 = selected_algorithm[1]
            if select_1 is not None:
                check[1] = st.checkbox(f'{selected_algorithm[1]}',key=select_1)
                if check[1]:
                    model_eval = st.radio(f'Select Model Evaluation Type for {select_1}',['None','Cross-Validation','Train-Test-Split'],index=0,horizontal=True)
                    scores_1 =train_model(task, algo_dict, select_1, X_train, y_train,train_scores,selects,model_evals,model_eval,val_scores,train_scoreds,val_scoreds)
                    #train_score,val_score = scores_1[0],scores_1[1]
            else:
                st.write('No Algorithms Selected for this box')
        with col_23[2]:
            select_2 = selected_algorithm[2]
            if select_2 is not None:
                check[2] = st.checkbox(f'{selected_algorithm[2]}',key=select_2)
                if check[2]:
                    model_eval = st.radio(f'Select Model Evaluation Type for {select_2}',['None','Cross-Validation','Train-Test-Split'],index=0,horizontal=True)
                    scores_2 = train_model( task, algo_dict, select_2, X_train, y_train,train_scores,selects,model_evals,model_eval,val_scores,train_scoreds,val_scoreds)
                    #train_score,val_score = scores_2[0],scores_2[1]
            else:
                st.write('No Algorithms Selected for this box')
        with col_23[3]:
            select_3 = selected_algorithm[3]
            if select_3 is not None:
                check[3] = st.checkbox(f'{selected_algorithm[3]}',key=select_3)
                if check[3]:
                    model_eval = st.radio(f'Select Model Evaluation Type for {select_3}',['None','Cross-Validation','Train-Test-Split'],index=0,horizontal=True)
                    scores_3 =train_model( task, algo_dict, select_3, X_train, y_train,train_scores,selects,model_evals,model_eval,val_scores,train_scoreds,val_scoreds)
                    
                    #train_score,val_score = scores_3[0],scores_3[1]
                
            else:
                st.write('No Algorithms Selected for this box')
        with col_24[0]:
            result = pd.DataFrame(
                    
                {
                    'model':selects,
                    'train_scores':train_scores,
                    'validation_scores':val_scores,
                    'model_evaluation':model_evals,
                    'conclusion':['Generalizing']*len(selects),
                    'train_score':train_scoreds,
                    'val_score':val_scoreds
                 
                    
                }
                )
            if task=='Classification':
                mask_1 =  ((result['train_scores']-result['validation_scores'])>0.2) 
            else:
                mask_1 =  np.abs((result['train_score']-result['val_score'])) >0.05
            if task=='Classification':
                mask_2 = (result['train_scores']>baseline_score) | (result['validation_scores']>baseline_score)
            else:
                mask_2 = (result['train_scores']<baseline_score) | (result['validation_scores']<baseline_score)
            
            result['conclusion'].mask(mask_1,'Overfitting',inplace=True)
            result['conclusion'].where(mask_2,'Underfitting',inplace=True)
            result.drop(['train_score','val_score'],axis=1,inplace=True)
            result.sort_values(by=['train_scores'],inplace=True,ascending=False)
            result.reset_index(inplace=True,drop=True)
            if (len(selects)>0):
                st.subheader('A Table Showing the Training and Validation Scores and Model Evaluation Method for each Model')
                st.dataframe(result,width=700)
            
            with col_24[1]:
                if (len(selects)>0):
                    st.subheader('A plot Showing the Training and Validation Scores for each Model')
                    fig = px.bar(data_frame=result,x='model',y=['train_scores','validation_scores'],barmode='group',hover_data=['model_evaluation'])
                    st.plotly_chart(fig)

            
    
#baseline setting
# model selection
# hyper parameter tuning
    