import pandas as pd
import numpy as np
def import_file(filename,delimiter):
    df = pd.read_csv(filename,delimiter=delimiter,low_memory=False)
    return df
#@st.cache(suppress_st_warning=True)
def info(df,include=None):
    shape = df.shape
    describe= df.describe()
    all_column = df.isnull().sum()>0#.columns#>(0.5*len(df))
    columns = all_column[all_column.values ==True].index.values
    # High Missing Values
    high_miss_column = df.isnull().sum()>(0.5*len(df))
    high_columns = high_miss_column[high_miss_column.values ==True].index.values
    high_columns
    return shape,describe,columns,high_columns

#@st.cache(suppress_st_warning=True)
def drop(df,columns):
    df = df.drop(columns,axis=1)
    return df
#@st.cache(allow_output_mutation=True)
def row_drop_na(df,columns,return_ind=False):
    na_index = df[np.any(df.loc[:,columns].isna(),axis=1)].index
    if return_ind:
        return na_index
    df.drop(na_index,axis=0,inplace=True)
    return df
#@st.cache(suppress_st_warning=True)
def mean_fill(df,columns):
    df.filen
#@st.cache(suppress_st_warning=True)
def cardinal_info(df):
    cat_df =df.select_dtypes(include='object')
    high_cardinal_columns = (cat_df.nunique()>(0.1*len(df)))
    high_cardinal_column = high_cardinal_columns[high_cardinal_columns.values==True].index.values
    return high_cardinal_column

'''def reg_fit(train_score,val_score):
    diff = train_score-val_score
    perc = 0.01*train_score
    if diff > perc and:
        return True'''