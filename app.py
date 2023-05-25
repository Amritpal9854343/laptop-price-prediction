import streamlit as st
import numpy as np
import pandas as pd
import pickle

# import the Model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))


st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['Company'].unique())
st.write(company)
# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())
st.write(type)
# Ram
Ram = st.selectbox('Ram(in GB)',df['Ram'].unique())
# Weight
weight = st.number_input('Weight of the laptop')
# touch_screen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])
# IPS 
ips = st.selectbox('IPS',['No','Yes'])
# screen-size
screen_size = st.number_input('Screen Size')
# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
# cpu
cpu = st.selectbox('CPU',df['Cpu Brand'].unique())
# HDD
hdd = st.selectbox('HDD(in GB)',[0,128,512,256,1024,2048])
# SDD
ssd = st.selectbox('SDD(in GB)',[0,128,512,256,1024])
# Brand
gpu = st.selectbox('GPU',df['Gpu Brand'].unique())
# OS
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    #query
    ppi = None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
    
    if ips=='Yes':
        ips=1
    else:
        ips=0
    
    x_res =  int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5/screen_size

    query = np.array([company,type,Ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    st.title("The Predicted Price is: " +  str(round(np.exp(pipe.predict(query)[0]),3)))