# Importing the necessary packages
import numpy as np
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')
# Setting up the streamlit page
st.title(":blue[PREDICTING SINGAPORE RESALE FLAT PRICES]")
with st.sidebar:
    selected = option_menu('Menu',["About","Predict Resale Price"])

if selected=="About":
    st.write("Here, we have considered the data of resale flat transactions from 1990 to present and used a machine learning regression model to make predictions on the resale price based on the factors like floor area, lease commence date, year, town, flat type, storey range, and flat model.")
    
# Load the required data
sfr_comb = pd.read_csv('Singapore_flat_data.csv')
TOWN = tuple(sfr_comb.town.unique())
FLAT_TYPE = tuple(sfr_comb.flat_type.unique())
FLAT_MODEL = tuple(sfr_comb.flat_model.unique())
STOREY_RANGE = tuple(sfr_comb.storey_range.unique())

# Load the regression model
with open('rand_resale.pkl','rb') as ranfor:
    model_rand = pickle.load(ranfor)

# Load the scaler
with open('scale_resale.pkl','rb') as scal:
    scale = pickle.load(scal)
    
# Load the one hot encoder
with open('one_hot_resale.pkl','rb') as oh:
    one_hot = pickle.load(oh)
    
if selected=="Predict Resale Price":
    col1,col2 = st.columns(2)
    with col1:
        flor_area = st.text_input(':green[**Enter Floor Area**]',key='flor_area')
        leas_comm_dat = st.text_input(':green[**Enter Lease Commence Date**]',placeholder='Eg:1976',key='lcm')
        year = st.text_input(':green[**Enter Year**]',placeholder='Eg:1998',key='yr')
        town = st.selectbox(':green[**Select Town**]',TOWN,index=None,key='town_r',placeholder='Select one')
    with col2:
        flat_typ = st.selectbox(':green[**Select Flat Type**]',FLAT_TYPE,index=None,key='fl_ty',placeholder='Select one')
        flat_mod = st.selectbox(':green[**Select Flat Model**]',FLAT_MODEL,index=None,key='fl_mod',placeholder='Select one')
        stor_rang = st.selectbox(':green[**Select Storey Range**]',STOREY_RANGE,index=None,key='st_ra',placeholder='Select one')
        if st.button(':red[Get Resale Price]'):
            X_res = np.array([[float(flor_area),float(leas_comm_dat),float(year),town,flat_typ,stor_rang,flat_mod]])
            X_res_oh = one_hot.transform(X_res[:,[3,4,5,6]])
            X_res1 = np.concatenate((X_res[:,[0,1,2]],X_res_oh),axis=1)
            X_res2 = scale.transform(X_res1)
            Res_pr = model_rand.predict(X_res2)
            st.write(f'{Res_pr[0]:.2f} SGD')
        
    
    
                         
    