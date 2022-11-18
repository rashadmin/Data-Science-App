import streamlit as st
from PIL import Image

st.title('Data Science Application Documentation')

st.markdown('### It is an application that allows you to interact and understand the basic of data science, it includes the following Functionalities:')
st.markdown('- ##### It introduces you to the concept of exploratory data analysis')
st.markdown('- ##### It also Introduces the basic method for treating missing values')
st.markdown('- ##### Splitting the data into the train and test set')
st.markdown('- ##### It transforms the data into number as our machine learning algorithm doesnt accept text or object datatype')      
st.markdown('- ##### Setting the baseline ; setting the baseline for regression is different from that of classifcation task')    
st.markdown('- ##### Selecting models to use for the prediction with either the conventional train-test-split or the more appropriate method cross validation.')

st.markdown('''### Starting from the homepage, Users are allowed to upload their own dataset in csv format or could as well as use the dataset that comes with the app for demo. I'd be using the datasets that comes with the app for explaining how the app works.''')
select_dataset_image = Image.open('img/select_dataset.png')
with st.expander('See Image'):
    st.image(select_dataset_image)
st.success('Only DataSet in .csv format is Accepted for now')
st.markdown(" #### A). Selecting the DataSet")
st.write(' ##### First we select dataset we want to use, there are four available dataset for demo, we would use the "crash_data.csv" dataset')
select_dataset_image_1 = Image.open('img/select_dataset_1.png')
with st.expander('See Image'):
    st.image(select_dataset_image_1)
st.markdown(' #### Getting to What we can do with the dataset, We have three options to pick from')
show_data_image_1 = Image.open('img/show_data.png')
with st.expander('See Image'):
    st.image(show_data_image_1)
st.markdown('##### The first shows the data information including the number of missing values,the description of the dataframe and the features with missing value advisable to drop.')
show_data_info_image_1 = Image.open('img/show_data_info.png')
with st.expander('See Image'):
    st.image(show_data_info_image_1)
st.write(' ##### Next, we move to the Exploratory Data Analysis Button to Check and do Some Data Analysis')
st.error('Note: Do ensure you pick the right variable for each data type as it would cause the plot to give a bad plot.')
eda_select_var = Image.open('img/eda_select_var.png')
with st.expander('See Image'):
    st.image(eda_select_var)
st.write(' ##### The next tab which is the numerical tab allows us to select the continuous and discrete data and also features we want to ignore')
num_select_var = Image.open('img/num_sel.png')
with st.expander('See Image'):
    st.image(num_select_var)
st.write(' ##### The next tab which is the numerical tab allows us to select the Nominal and Ordinal data and also features we want to ignore')
cat_select_var = Image.open('img/cat_sel.png')
with st.expander('See Image'):
    st.image(cat_select_var)
st.markdown(' ##### The other tabs helps us visualize the relationship between;') 
st.markdown('- ###### The Numerical and Catergorical data')
st.markdown('- ###### The Numerical and other Numerical data')
st.markdown('- ###### The Categorical and other Categorical Data')
st.markdown('- ###### We can also check for the Relationship  between a Categorical data and the Target data.')
st.markdown('- ###### We can also check for the Relationship  between a Numerical data and the Target data.')

st.write(' ##### Next, we move to the Data Preprocessing and Modelling Button to Check and do Some Data Modelling')
data_modelling_image_1 = Image.open('img/data_modelling.png')
with st.expander('See Image'):
    st.image(data_modelling_image_1)
st.write(' ##### We will be starting with the Dropping tab that allows us to drop columns we will not be needing, we have three instances : ')
dropping_image_1 = Image.open('img/dropping.png')
with st.expander('See Image'):
    st.image(dropping_image_1)
st.markdown('- ###### Drop High and Low Cardinality Features : This gives us the option to drop Feature with single Values or Feature with different values and is categorical')
st.markdown('- ###### Drop Feature with High Percentage of Missing Values : The Feature marked red in the Data Information tab are printed out and users are given to drop them, edit the feature to drop or ignore them ')
st.markdown('- ###### Drop Leaky,Multicollinear and Features not needed : There could be Feature that leaks information about the target which is bad for our model, or target which are highly correlated with each other, we would need to drop them')

st.write(' ##### The Next tab is the Splitting tab that allows user to specify their target data and also the Machine learning task our model would be doing (i.e Regression or Classification)')
splitting_image_1 = Image.open('img/splitting_1.png')
with st.expander('See Image'):
    st.image(splitting_image_1)
st.write(' ##### The next tab is called Filling, it allows us to fill feature with mild amount of missing values with either the Mean, Median,or the Most Frequent Values, we could also drop the rows with missing Values for each Features.')
filling_image_1 = Image.open('img/filling.png')
with st.expander('See Image'):
    st.image(filling_image_1)
st.write(' ##### The next tab is Transforming, It allows us to Transform our already preprocess data either through One Hot Encoding or Ordinal Encoding.')
transform_image_1 = Image.open('img/transforming.png')
with st.expander('See Image'):
    st.image(transform_image_1)
st.write(' ##### The Next Tab which is the Baseline Tab allows Users to select the Model they want to use, the Models available are :')
baseline_image_1 = Image.open('img/baseline.png')
with st.expander('See Image'):
    st.image(baseline_image_1)
st.markdown('-  ###### Regression : Linear Regression, Decision Tree Regressor,Random Forest Regressor and Xgboost Regressor')
st.markdown('-  ###### Classification : Logistic Regression, Decision Tree Classifier,Random Forest Classifier and Xgboost Classifier')

st.write(' ##### Finally we are at the Modelling Tab that allows us to Model the data using the Specified Model in the Baseline Tab, Each Model has two Method of Evaluatio  which are the Train-Test-Split and the Cross Validation Method')
model_image_1 = Image.open('img/modelling.png')
with st.expander('See Image'):
    st.image(model_image_1)
st.write(' ##### The Evaluation Method selected trains the Model using the Method and Appends to the Table that shows the Model Name, Test Score, Train Score, and also the Method of Evaluation,  A bar chart is also plotted that compared the train and test score for all trained model ')
result_image_1 = Image.open('img/result.png')
with st.expander('See Image'):
    st.image(result_image_1)
