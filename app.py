import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open('loan_model.pkl','rb'))

@st.cache(allow_output_mutation=True)
def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def new_dataframe(new_data):
    loan=pd.DataFrame(new_data,columns=['Age','Experience','Income','Family'])
    return loan

header_content = st.container()
prediction = st.container()
dataset_descrb = st.container()
#main


with header_content:
    st.title('Hello Friends ! , This is an End to End to Machine Learning Project :smile:')
    first_para = '<p style="font-family:Courier; color:Black; font-size: 20px;">In this project I have developed a ML model to check if the personal loan available or not for the customer . By using the Age , Experience , Income , Family members count . Lets predit it .....</p>'
    st.markdown(first_para, unsafe_allow_html=True)

    st.header('Objective:')
    st.text('To predict whether a liability customer will get a personal loan or not.')

with dataset_descrb:
    st.header('*Personal Loan dataset*')
    second_para = '<p style="font-family:Courier; color:Black; font-size: 20px;">Lets see the insights of the dataset by visualizing the dataset...</p>'
    st.markdown(second_para, unsafe_allow_html=True)
    df = load_data("Bank_Personal_Loan_Modelling.csv")
    load_data = ['Experience','Income']
    features = df[load_data]
    d=features.head(50)
    st.line_chart(d)
    load_data1= ['Age','Experience']
    features = df[load_data1]
    d2=features.head(50)
    st.line_chart(d2)
    with st.expander("See explanation"):
     st.write("""
         The above chart is derived using the below given data set...
     """)
     st.dataframe(df)

with prediction:
    st.header(""" Let's predict""")
    third_para = '<p style="font-family:Courier; color:Black; font-size: 20px;">Enter the values below and lets check whether a liability customer will get a personal loan or not</p>'
    st.markdown(third_para, unsafe_allow_html=True)
    a, b = st.columns(2)
    age = a.text_input('Enter your age (years):',0)
    age = int(age)
    Experience = b.text_input('Enter your Experience (years):', 0)
    Experience = int(Experience)
    Income = a.text_input('Enter your Annual income (In Thousands dollars):', 0)
    Income = int(Income)
    Family = b.slider('Enter your Family members count :',min_value=0, max_value=5, value=1, step=1)
    new_data = [[age,Experience,Income, Family]]
    new_df1=new_dataframe(new_data)
    predict_value = model.predict(new_df1)
    result = st.button("Predict")
    if result:
        if predict_value == 1:
            st.subheader('I am very happy to say that customer is able to apply for personal loan :smile:')
        else:
            st.subheader('Sorry I am sad to say that customer is not able to apply for personal loan :pensive:')