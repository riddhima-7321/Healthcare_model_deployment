import numpy as np
from flask import Flask, request, render_template
import pickle
import requests
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    Age=(request.form["AGE"])
    Gender=request.form["gender"]
    BloodType=request.form["BloodType"]
    MedicalCondition=request.form["MEDICAL CONDITION"]
    InsuranceProvider=request.form["INSURANCE PROVIDER"]
    BillingAmount=(request.form["billing amount"])
    RoomNumber=(request.form["room number"])
    AdmissionType=request.form["Admission Type"]
    Medication=request.form["Medication"]
    Duration=(request.form["duration"])

    # if (Gender == "Male"):
    #     Gender=0
    # if (Gender == "Female"):
    #     Gender=1

    # if(BloodType == "O-"):
    #     BloodType=0
    # if(BloodType == "O+"):
    #     BloodType=1
    # if(BloodType == "B-"):
    #     BloodType=2
    # if(BloodType == "AB+"):
    #     BloodType=3
    # if(BloodType == "A+"):
    #     BloodType=4
    # if(BloodType == "AB-"):
    #     BloodType=5
    # if(BloodType == "A-"):
    #     BloodType=6
    # if(BloodType == "B+"):
    #     BloodType=7

    # if(MedicalCondition == "Diabetes"):
    #     MedicalCondition=0
    # if(MedicalCondition == "Asthma"):
    #     MedicalCondition=1
    # if(MedicalCondition == "Obesity"):
    #     MedicalCondition=2
    # if(MedicalCondition == "Arthritis"):
    #     MedicalCondition=3
    # if(MedicalCondition == "Hypertension"):
    #     MedicalCondition=4
    # if(MedicalCondition == "Cancer"):
    #     MedicalCondition=5
    
    # if(InsuranceProvider == "Medicare"):
    #     InsuranceProvider=0
    # if(InsuranceProvider == "UnitedHealthcare"):
    #     InsuranceProvider=1
    # if(InsuranceProvider == "Aetna"):
    #     InsuranceProvider=2
    # if(InsuranceProvider == "Cigna"):
    #     InsuranceProvider=3
    # if(InsuranceProvider == "Blue Cross"):
    #     InsuranceProvider=4

    # if(Medication == "Aspirin"):
    #     Medication=0
    # if(Medication == "Lipitor"):
    #     Medication=1
    # if(Medication == "Penicillin"):
    #     Medication=2
    # if(Medication == "Paracetamol"):
    #     Medication=3
    # if(Medication == "Ibuprofen"):
    #     Medication=4

    # if(AdmissionType == "Elective"):
    #     AdmissionType=0
    # if(AdmissionType == "Emergency"):
    #     AdmissionType=1
    # if(AdmissionType == "Urgent"):
    #     AdmissionType=2
    
    features=np.array([int(Age),int(Gender),int(BloodType),int(MedicalCondition),int(InsuranceProvider),int(BillingAmount),int(RoomNumber),int(AdmissionType),int(Medication),int(Duration)])
    features = features.reshape(1, -1)
    output = model.predict(features)
    output=int(output[0])
    if(output==0):
        output="Inconclusive"
    if(output==1):
        output="Normal"
    if(output==2):
        output="Abnormal"
    return render_template('index.html', pred='TEST RESULTS:{}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)