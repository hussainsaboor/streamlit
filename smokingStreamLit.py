import streamlit as st
import pandas as pd
import joblib

st.title('Smoking App \n\n')

parameter_input_values=[]
parameter_list=['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'eyesight(left)',
       'eyesight(right)', 'hearing(left)', 'hearing(right)', 'systolic',
       'relaxation', 'fasting blood sugar', 'Cholesterol', 'triglyceride',
       'HDL', 'LDL', 'hemoglobin', 'Urine protein', 'serum creatinine', 'AST',
       'ALT', 'Gtp', 'dental caries', 'smoking']

with st.spinner('Fetching Latest ML Model'):
    # Use pickle to load in the pre-trained model
    model = joblib.load("TrainModel\smokingModel.pkl")
    st.success('Model has been loaded!')

age_number = st.number_input("Age",min_value=20, max_value=80, value=42,key=1)
height_number = st.number_input('Height(cm)',min_value=140.00, max_value=190.00, value=166.00, key=2)
weight_number = st.number_input('Weight(Kg)',min_value=45.0, max_value=100.0, value=66.0,key=3)
waist_number = st.number_input('Waist(cm)',min_value=59.000, max_value=108.000, value=81.351,key=4)
left_eyesight_number = st.number_input('Left Eyesight',min_value=0.100, max_value=1.500, value=1.002,key=5)
right_eyesight_number = st.number_input('Right Eyesight',min_value=0.100, max_value=1.500, value=0.997,key=6)
left_hearing_number = st.number_input('Left hearing',min_value=1.00, max_value=2.00, value=1.02,key=7)
right_hearing_number = st.number_input('Right hearing',min_value=1.00, max_value=2.00, value=1.02,key=8)
systolic_number = st.number_input('Systolic',min_value=95.00, max_value=160.00, value=120.86,key=9)
relaxation_number  = st.number_input('Relaxation',min_value=52.00, max_value=96.00, value=75.28,key=10)
fastingBloodSugar_number = st.number_input('fasting blood sugar',min_value=70.0, max_value=188.0, value=97.0,key=11)
cholesterol_number = st.number_input('Cholesterol',min_value=116.00, max_value=322.00, value=197.72,key=12) 
triglyceride_number = st.number_input('Triglyceride',min_value=47.00, max_value=318.00, value=122.77,key = 13)
HDL_number = st.number_input('HDL',min_value=30.00, max_value=102.00, value=57.27,key=14)
LDL_number = st.number_input('LDL',min_value=43.00, max_value=226.00, value=115.65,key =15)
hemoglobin_number = st.number_input('hemoglobin',min_value=11.600, max_value=17.900, value=14.763,key =18)
UrineProtein_number = st.number_input('Urine protein',min_value=1.00, max_value=4.00, value=1.07,key =19)
SerumCreatinine_number =st.number_input('serum creatinine',min_value=0.500, max_value=1.300, value=0.871,key=20)
AST_number =st.number_input('AST',min_value=13.00, max_value=60.00, value=24.23,key=21)
ALT_number =st.number_input('ALT',min_value=9.00, max_value=114.00, value=25.51,key=22)
Gtp_number =st.number_input('Gtp',min_value=8.00, max_value=305.00, value=37.39,key=23)
DentalCaries_number=st.number_input('Dental caries',min_value=0.00, max_value=1.00, value=0.22,key=24)
smoking_number=st.number_input('smoking',min_value=0.00, max_value=1.00, value=0.43,key=25)

parameter_input_values=[age_number,height_number,weight_number,waist_number,left_eyesight_number,right_eyesight_number,left_hearing_number,right_hearing_number,
systolic_number,relaxation_number,fastingBloodSugar_number,cholesterol_number,triglyceride_number,HDL_number,LDL_number,hemoglobin_number,
UrineProtein_number,SerumCreatinine_number,AST_number,ALT_number,Gtp_number,DentalCaries_number,smoking_number]

parameter_dict=dict(zip(parameter_list, parameter_input_values))

st.write('\n','\n')
st.title('Your Input Summary')

st.write(parameter_dict)

st.write('\n','\n')

def predict(input_predict):
    values = input_predict['data'] 

    input_variables = pd.DataFrame([values],
                                columns=parameter_list, 
                                dtype=float,
                                index=['input'])    
    
    # Get the model's prediction
    prediction = model.predict(input_variables)
    print("Prediction: ", prediction)
    prediction_proba = model.predict_proba(input_variables)[0][1]
    print("Probabilities: ", prediction_proba)

    ret = {"prediction":float(prediction),"prediction_proba": float(prediction_proba)}
    
    return ret

if st.button("Click Here to Predict"):

    PARAMS={'data':list(parameter_dict.values())}
    
    r=predict(PARAMS)
    
    st.write('\n','\n')
    
    prediction_proba=r.get('prediction_proba')
    prediction_proba_format = str(round(float(r.get('prediction_proba')),1)*100)+'%'
    
    prediction_value=r.get('prediction')
    
    prediction_bool='Positive' if float(prediction_proba) > 0.4 else 'Negative'
    
    st.write(f'Your Smoking Prediction is: **{prediction_bool}** with **{prediction_proba_format}** confidence')
