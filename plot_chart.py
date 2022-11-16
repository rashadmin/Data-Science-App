import plotly.express as px
import streamlit as st

def plot_discrete_ordinal(dis_col_plot,df,key):
    plot_type = st.radio('Select Discrete Plot Type',options=['barplot','pieplot'],horizontal=True,key=key)
    if plot_type =='barplot':
        data = df[dis_col_plot].value_counts()
        fig = px.bar(data)
    else:
        names = df[dis_col_plot].value_counts().index
        values = df[dis_col_plot].value_counts().values
        fig = px.pie(data_frame=df,names=names,values=values,labels=names)
    return fig

def plot_continuous(cont_col_plot,df):
    plot_type = st.radio('Select Plot Type',options=['histplot','boxplot'],horizontal=True)
    if plot_type =='histplot':
        data = df[cont_col_plot]
        fig = px.histogram(data)
    else:
        data = df[cont_col_plot]
        fig = px.box(data,orientation='h')
    return fig

def nvn_relationship_plot(df,dis_1,dis_2,num_col_1_plot,num_col_2_plot):
    col_40 = st.columns([5,35,5,45,5])
    with col_40[1]:
        st.subheader(f'A Scatter Plot of the Relationship Between {num_col_1_plot} and {num_col_2_plot}')
        fig = px.scatter(data_frame=df,x=num_col_1_plot,y=num_col_2_plot)
        st.plotly_chart(fig)
    if dis_1 and not dis_2:
        data = df.groupby(num_col_1_plot)[num_col_2_plot].mean()
    elif dis_2 and not dis_1:
        data = df.groupby(num_col_2_plot)[num_col_1_plot].mean()
    elif dis_1 and  dis_2:
        data = df.groupby(num_col_2_plot)[num_col_1_plot].value_counts().unstack()
    if dis_1 or dis_2:
        with col_40[3]:
            st.subheader(f'A Bar Plot of the Relationship Between "{num_col_1_plot}" and "{num_col_2_plot}"')
            fig = px.bar(data,width=800)
            st.plotly_chart(fig)
def nvc_relationship_plot(df,dis_1,n_col_1_plot,c_col_1_plot,num_miss=False,cat_miss=False):
    col_41 = st.columns([5,35,5,45,5])
    with col_41[1]:
        st.subheader(f'A Box Plot of the Relationship Between "{n_col_1_plot}" and "{c_col_1_plot}"')
        fig = px.box(data_frame=df,x=n_col_1_plot,y=c_col_1_plot,orientation='h')
        st.plotly_chart(fig)
    if dis_1:
        data = df.groupby(c_col_1_plot)[n_col_1_plot].value_counts().unstack()
    else:
        data = df.groupby(c_col_1_plot)[n_col_1_plot].mean()

    with col_41[3]:
        st.subheader(f'A Bar Plot of the Relationship Between "{n_col_1_plot}" and "{c_col_1_plot}"')
        fig = px.bar(data,width=800,barmode='group')
        st.plotly_chart(fig)
        
def cvc_relationship_plot(df,cat_col_1_plot,cat_col_2_plot):
    col_42 = st.columns([5,35,5,45,5])
    with col_42[1]:
        st.subheader(f'A Stacked Bar Plot of the Relationship Between "{cat_col_1_plot}" and "{cat_col_2_plot}"')
        data = df.groupby(cat_col_1_plot)[cat_col_2_plot].value_counts().unstack()
        fig = px.bar(data,width=800,)
        st.plotly_chart(fig)
    

    with col_42[3]:
        st.subheader(f'A Side by Side Bar Plot of the Relationship Between "{cat_col_1_plot}" and "{cat_col_2_plot}"')
        data = df.groupby(cat_col_1_plot)[cat_col_2_plot].value_counts().unstack()
        fig = px.bar(data,width=800,barmode='group')
        st.plotly_chart(fig)
def feat_target_plot(df,num_col_tar_plot,cat_col_tar_plot,task_plot,target,num_miss=False,cat_miss=False):
    col_43 = st.columns([5,35,20,35,5])
    if task_plot == 'Regression':
        with col_43[1]:
            if not cat_miss:
                type_plot = st.radio('',options=['box_plot','bar_plot'],horizontal=True)
                if type_plot== 'bar_plot':
                    st.subheader(f'A Bar Plot of the Relationship Between "{cat_col_tar_plot}" and Target "{target}"')
                    data = df.groupby(cat_col_tar_plot)[target].mean()
                    cat_fig = px.bar(data)
                    st.plotly_chart(cat_fig)
                else:
                    st.subheader(f'A Box Plot of the Relationship Between "{cat_col_tar_plot}" and "{target}"')
                    fig = px.box(data_frame=df,x=cat_col_tar_plot,y=target,orientation='h')
                    st.plotly_chart(fig)
            else:
                st.write('No Feature Available to Plot')
        with col_43[3]:
            if not num_miss:
                st.header('')
                st.write(' ')
                st.subheader(f'A Scatter Plot of the Relationship Between "{num_col_tar_plot}" and Target "{target}"')
                num_fig = px.scatter(data_frame=df,x=num_col_tar_plot,y=target)
                st.plotly_chart(num_fig)
            else:
                st.write('No Feature Available to Plot')
    else:
        with col_43[1]:
            if not cat_miss:
                st.header('')
                st.write(' ')
                st.subheader(f'A Bar Plot of the Relationship Between "{cat_col_tar_plot}" and Target "{target}"')
                data = df.groupby(cat_col_tar_plot)[target].value_counts().unstack()
                fig = px.bar(data,width=800,barmode='group')
                st.plotly_chart(fig)
        with col_43[3]:
            if not num_miss:
                type_plot = st.radio('',options=['box_plot','bar_plot'],horizontal=True)
                if type_plot== 'bar_plot':
                    st.subheader(f'A Box Plot of the Relationship Between "{num_col_tar_plot}" and Target "{target}"')
                    data = df.groupby(target)[num_col_tar_plot].mean()
                    fig = px.bar(data)
                    st.plotly_chart(fig)
                else:
                    
                    st.subheader(f'A Box Plot of the Relationship Between "{num_col_tar_plot}" and "{target}"')
                    fig = px.box(data_frame=df,x=num_col_tar_plot,y=target,orientation='h')
                    st.plotly_chart(fig)
            
    
    
def target_plot(task_plot,df,target):
    col_55 = st.columns([5,40,10,40,5]) 
    if task_plot == 'Classification':
        with col_55[1]:
            st.subheader('A BarPlot Showing the Count of the Target Variable')
            data = df[target].value_counts()
            fig_bar = px.bar(data)
        with col_55[1]:
            st.plotly_chart(fig_bar)
        with col_55[3]:
            st.subheader('A Pie Chart Showing the Count of the Target Variable')
            names = df[target].value_counts().index
            values = df[target].value_counts().values
            fig_pie = px.pie(data_frame=df,names=names,values=values,labels=names)
        with col_55[3]:
            st.plotly_chart(fig_pie)
    else:
        with col_55[1]:
            st.subheader('A Histogram Plot Showing the Count of the Target Variable')
            data = df[target]
            fig_hist = px.histogram(data)
        with col_55[1]:
            st.plotly_chart(fig_hist)
        with col_55[3]:
            st.subheader('A Box Plot Showing the Distribution of the Target Variable')
            fig_box = px.box(data)
        with col_55[3]:
            st.plotly_chart(fig_box)