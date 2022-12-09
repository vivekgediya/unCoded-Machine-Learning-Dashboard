import streamlit as st 
from streamlit_pandas_profiling import st_profile_report
st.set_option('deprecation.showPyplotGlobalUse', False)

import numpy as np
import pandas as pd
import pandas_profiling
from math import sqrt

from PIL import Image

import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

## In-built Dataset
from sklearn import datasets

## Handling Missing Data
from sklearn.impute import SimpleImputer

## Scaling
from sklearn.preprocessing import StandardScaler

## Model selection (Data Splitting) 
from sklearn.model_selection import train_test_split

## PCA
from sklearn.decomposition import PCA

## Classification Algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Ensemble Classification Model
from sklearn.ensemble import RandomForestClassifier

## Regression Algorithms

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn import linear_model

# Ensemble Regression Model
from sklearn.ensemble import RandomForestRegressor

## Evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix 

# ------------------------------------------------- " Main Code " ----------------------------------------------------------------

st.title("unCoded ML Dashboard ✨")
st.write("""#### **Simple Machine Learning Web Application using Streamlit** """)

image = Image.open('Images\\banner-ml.jpg')
st.image(image ,use_column_width=True)

st.write("""### **Let's Explore different types of Supervised ML Algorithms**""")

def main():

    st.sidebar.write('# Project Member ')
    st.sidebar.write('### Vivekkumar Gediya (MCS22011)')
    st.sidebar.write('### Raman Singh (MCS22015)')
    st.sidebar.write('-------------------------------')

    st.sidebar.write('## Select appropriate ML Life Cycle stage from below')
    activities = ['EDA', 'Profiling Report', 'Visualization', 'Feature Engineering & Model Building']
    option = st.sidebar.selectbox('Select Option',activities)

    data = st.file_uploader('Upload Dataset',type=['csv'])
    if data is not None:
            data.seek(0)
            df = pd.read_csv(data)
            st.dataframe(df.head(50))
            st.success('Data uploaded & Loaded successfully')
    st.write('-----------------------------------------------------------------------------------------')


    # """""""""""""""""""""""""""""""""" EDA Operations """"""""""""""""""""""""""""""""""

    if option=='EDA':
        st.subheader('Exploratory Data Analysis')
        
        if st.checkbox('Display Shape'):
            st.write(df.shape)
        if st.checkbox('Display columns'):
            st.write(df.columns)
        if st.checkbox('Display Summary'):
            st.write(df.describe().T)
        if st.checkbox('Display Null Values'):
            st.write(df.isnull().sum())
        if st.checkbox('Display Data Types'):
            st.write(df.dtypes)
        if st.checkbox('Display Correlations of Columns'):
                st.write(df.corr())

    
    if option == "Profiling Report": 
        if st.checkbox('Generate a report'):
            st.subheader("Automated Profile Report")
            profile_df = df.profile_report()
            profile_df.to_file(output_file='Report_Profiling.html')
            st_profile_report(profile_df)
            st.success('Report saved successfully on your machine')          


    # # # """""""""""""""""""""""""""""""""" Data Preprocessing """"""""""""""""""""""""""""""""""

    # # if option=='Data Preprocessing':
    # #     st.subheader('Data Preprocessing')

    #     if st.checkbox('Display Null Values'):
    #         st.write(df.isnull().sum())
        
    #     st.info('###### *If data have null value then only perform below operation')

    #     if st.checkbox('Remove Null value'):   
    #         all_columns = df.columns.to_list()  
    #         input = st.multiselect('Select columns that conains null value',all_columns)
    #         df1 = df[input]
    #         st.write(df1)
    #         #st.dataframe(df[input])

    #         if st.checkbox("Check the box to Replace with mean"):
    #             imputer = SimpleImputer(missing_values=np.nan, strategy='mean',copy=False)
    #             imputer = imputer.fit(df1)
    #             df1 = imputer.transform(df1)
    #             df1 = pd.DataFrame(df1)
    #             st.write(type(df1))
    #             st.dataframe(df1)
    #             df = pd.concat([df, df1],axis=1)
    #             df = pd.DataFrame(df)
    #             st.dataframe(df) 
    #     df = pd.DataFrame(df)       

            
    # """""""""""""""""""""""""""""""""" Visualization Methods """"""""""""""""""""""""""""""""""
    if option=='Visualization':
        st.subheader("Data Visualization")

        if st.checkbox('Select Multiple Columns to plot *'):
            selected_columns = st.multiselect('Select Columns',df.columns)
            df1 = df[selected_columns]
            st.dataframe(df1)

        if st.checkbox('Disply Heatmap'):
            st.info('Please select multiple columns first to plot pairplot')
            st.write(sns.heatmap(df1.corr(),linewidths=.5,vmax=1,annot=True,cmap='viridis',square=True))
            st.pyplot()
        
        if st.checkbox('Display Pairplot'):
            st.info('Please select multiple columns first to plot pairplot')
            st.write(sns.pairplot(df1,diag_kind='kde'))
            st.pyplot()

        if st.checkbox('Display Pie chart'):
            all_columns = df.columns.to_list()
            pie_columns = st.selectbox('Select columns to dispaly pie chart',all_columns)
            piechart = df[pie_columns].value_counts().plot.pie(autopct='%1.1f%%')
            st.write(piechart)
            st.pyplot()

        if st.checkbox('Display Boxplot'):
            sns.boxplot(data=df)
            st.pyplot()


    # """""""""""""""""""""""""""""""""" Feature Engineering & Model Building """"""""""""""""""""""""""""""""""

    if option=='Feature Engineering & Model Building':

        # Null Values
        st.subheader('Remove Null Values*')

        if st.checkbox('Display Null Values*'):
            st.write(df.isnull().sum())
            df1 = pd.DataFrame(df)
        
        st.info('###### If data have null value then only perform below operation')

        if st.checkbox('Remove Null value'):   
            all_columns = df.columns.to_list()  
            input = st.multiselect('Select columns that conains null value',all_columns)
            df1 = df[input]
            st.write(df1)
            column_list = list(df1.columns)
            #st.dataframe(df[input])

            if st.checkbox("Check the box to Replace with mean"):
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean',copy=False)
                imputer = imputer.fit(df1)
                df1 = imputer.transform(df1)
                # column_list = list(df.columns)
                st.write(column_list)
                df1 = pd.DataFrame(df1,columns=column_list)
                st.write(type(df1))
                st.dataframe(df1)
                #df = pd.concat([df, df1],axis=1)
                #df = pd.DataFrame(df)
                # st.dataframe(df) 
        st.write('-------------------------------')

        # Model Selection
        st.subheader('Model Selection : Choose appropiate model for your data*')

        if st.checkbox('Select X and y from Columns*'):
            
            # X and y variables
            all_columns = df.columns.to_list()
            target = st.selectbox('Select columns that is Depended or Output attribute called as "y"',all_columns)
            y = df[target]
            st.dataframe(y)
            st.write('Target variable y is:',target)

            all_columns = df1.columns.to_list()
            input = st.multiselect('Select columns that is Independed or Input attribute called as "X"',all_columns)
            st.write('Input variable X are:')
            X = df1[input]
            st.dataframe(X)

        st.write('-------------------------------')

        st.subheader('Apply Feature Scaling')
        st.warning('Do not apply Scaling on Decision based algorithms')
        if st.checkbox('Scale'):
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
                # x_test = scaler.transform(x_test)
                # x_train = pd.DataFrame(x_train)
                # x_test = pd.DataFrame(x_test)
                # st.write(type(x_train))
                # st.write(type(x_test))
                # st.write(x_train)

        
        # Random State
        seed = st.sidebar.slider('Random State',0,200,0,10)
        st.sidebar.write('-------------------------------')

        # Sidebar - Classification or Regression problem
        st.sidebar.write('## Choose Algorithm from here')
        radio = st.sidebar.radio("Select the type of Supervised Problem,", ('Classification', 'Regression',))

        if radio == "Classification":
            algo_name = st.sidebar.selectbox('Select appropiate Classifier',('Logistic Regression','K-Nearest Neighbours','Support Vector Classifier','Naive Bayes Classifier','Decision Tree Classifier','Random Forest Classifier'))
         
        if radio == "Regression":
            algo_name = st.sidebar.selectbox('Select appropiate Classifier',('Linear Regression','Lasso Regression','Support Vector Regressor','Random Forest Regressor'))

        # HyperParameter of Algorithms
        def add_parameter(name_of_algo):
            params = dict()
            if name_of_algo == 'K-Nearest Neighbours': 
                K = st.sidebar.slider('K',1,20,1,2)
                params['K'] = K

            if name_of_algo == 'Support Vector Classifier': 
                C = st.sidebar.slider('C',0.01,15.0)
                params['C'] = C 
            
            if name_of_algo == 'Support Vector Regressor': 
                C = st.sidebar.slider('C',0.01,15.0)
                params['C'] = C 

            if name_of_algo == 'Lasso Regression': 
                alpha = st.sidebar.slider('alpha',0.01,15.0)
                params['alpha'] = alpha            
            
                
            return params

        params = add_parameter(algo_name) 

        # All Classification and Regression Algorithms
        def get_classifier(name_of_algo,params): 
            model = None

            # Classification Algorithm
            if name_of_algo == 'Logistic Regression':
                model = LogisticRegression()
            if name_of_algo == 'Support Vector Classifier': 
                model = SVC(C=params['C'])
            if name_of_algo =='K-Nearest Neighbours': 
                model = KNeighborsClassifier(n_neighbors=params['K'])
            if name_of_algo == 'Naive Bayes Classifier':
                model = GaussianNB()
            if name_of_algo == 'Decision Tree Classifier':
                model = DecisionTreeClassifier()
            if name_of_algo == 'Random Forest Classifier':
                model = RandomForestClassifier()
            # else:
            #     st.warning('Unknown classifier or choose from above')

            # Regression Algorithms
            if name_of_algo == 'Linear Regression':
                model = LinearRegression()
            if name_of_algo == 'Lasso Regression':
                model = linear_model.Lasso(alpha=params['alpha'])
            if name_of_algo == 'Support Vector Regressor':
                model = SVR(C=params['C'])
            if name_of_algo == 'Random Forest Regressor':
                model = RandomForestRegressor()
                
            return model

        model = get_classifier(algo_name,params) 

        st.subheader("Splitting The data")
        # Train_Test_Split
        if st.checkbox('Split data into train and test*'): 
            st.info("Please select Random State from the Side Menubar, Test size is 20%")
            x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=seed)
            st.success('Done!')

        # st.write('-------------------------------')
        # st.subheader('Apply Feature Scaling')
        # st.warning('Do not apply Scaling on Decision based algorithms')

        # if st.checkbox('Scale'):
        #     scaler = StandardScaler()
        #     x_train = scaler.fit_transform(x_train)
        #     x_test = scaler.transform(x_test)
        #     x_train = pd.DataFrame(x_train)
        #     x_test = pd.DataFrame(x_test)
        #     # st.write(type(x_train))
        #     # st.write(type(x_test))
        #     # st.write(x_train)
            
        st.write('-------------------------------')
        # Model fitting
        st.subheader('Model Fitting & Prediction*')
        if st.checkbox('Fit model'):
            model.fit(x_train,y_train)
            # st.write(x_train)
            st.success('Training Completed!')

        # Model Prediction
        if st.checkbox('Show predicted output'):
            y_pred = model.predict(x_test)
            st.write(y_pred)
            #st.write(y_test)

        st.write('-----------------------------------')

        # Metrics Evaluation
        if st.checkbox('Evaluation Metrics*'):

            if radio == "Classification":

                accuracy = accuracy_score(y_test,y_pred)
                st.write('Accuracy of ', algo_name,'classifier is ',accuracy)
                precisoin = precision_score(y_test, y_pred,average=None)
                st.write('Precision of ',algo_name,'classifier is ',precisoin)
                recall = recall_score(y_test, y_pred,average=None)
                st.write('Recall of ',algo_name,'classifier is ',recall)
                F1_score = f1_score(y_test, y_pred,average='macro')
                st.write('F1_Score of ',algo_name,'classifier is ',F1_score)

                if st.checkbox('Confusion Matrix'):
                    cm = confusion_matrix(y_test, y_pred)
                    # Transform to df for easier plotting
                    cm_df = pd.DataFrame(cm)

                    sns.heatmap(cm_df, annot=True)
                    plt.title("Confusion Matrix")
                    plt.ylabel('Actual label')
                    plt.xlabel('Predicted label')
                    st.pyplot()

            
            if radio == "Regression":
                
                mse = mean_squared_error(y_test,y_pred)
                st.write('RMSE of ', algo_name,' is ',sqrt(mse))
                mae = mean_absolute_error(y_test,y_pred)
                st.write('MAE of ', algo_name,' is ',mae)
                r2 = r2_score(y_test,y_pred)
                st.write('R2 Score of ', algo_name,' is ',r2)
                n, p = df.shape
                q, k = x_train.shape
                adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)
                st.write('Adjusted R2 Score of ', algo_name,' is ',adj_r2)
                if adj_r2 >= 0.45:
                    st.success("**It's considered to be Good Model**")

    st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write('');st.sidebar.write(''); 
    st.sidebar.write('## About')
    st.sidebar.write('This is interactive Web App for Machine learning.')
    if st.sidebar.button("Thank You!"):
        st.sidebar.balloons()
    
  
if __name__ == '__main__':
    main()