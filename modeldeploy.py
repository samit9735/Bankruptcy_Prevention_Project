#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

st.title('Model Deployment:Logistic Regression')
st.sidebar.header('user input parameters')
def user_input_features():
    industrial_risk=st.selectbox('Industrial_Risk',('0','0.5','1'))
    management_risk=st.selectbox('Management_Risk',('0','0.5','1'))
    financial_flexibility=st.selectbox('Financial_Flexibility',('0','0.5','1'))
    credibility=st.selectbox('Credibility',('0','0.5','1'))
    competitiveness=st.selectbox('Competitiveness',('0','0.5','1'))
    operating_risk=st.selectbox('Operating_Risk',('0','0.5','1'))
    
    data={'Industrial_Risk':industrial_risk,
          'Management_Risk':management_risk,
          'Financial_Flexibility':financial_flexibility,
          'Credibility':credibility,
          'Competitiveness':competitiveness,
          'Operating_Risk':operating_risk}
    features=pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
st.subheader('user Input Parameters')
st.write(df)

bank=pd.read_csv("D:\\project\\bankruptcy_prev.csv")
bank=bank.dropna()

x=bank.iloc[:,0:6]
y=bank.iloc[:,6]
clf=LogisticRegression()
clf.fit(x,y)
prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

st.subheader('predicted Result')
st.write("Non-Bankruptcy" if prediction_proba[0][1]>0.5 else "Bankruptcy")

st.subheader('Prediction Probability')
st.write(prediction_proba)

