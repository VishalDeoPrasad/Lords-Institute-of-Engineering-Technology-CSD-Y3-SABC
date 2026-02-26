from django.shortcuts import render
import pandas as pd
from django.conf import settings
import os
import joblib

model_path = os.path.join(settings.BASE_DIR, "ml_model", "loan_model.joblib")
encoder_path = os.path.join(settings.BASE_DIR, "ml_model", "encoders.joblib")
columns_path = os.path.join(settings.BASE_DIR, "ml_model", "columns.joblib")

model = joblib.load(model_path)
encoders = joblib.load(encoder_path)
columns = joblib.load(columns_path)

# Create your views here.
def home(request):
    if request.method == 'POST':
        Gender = request.POST['Gender']
        Married = request.POST['Married']
        Dependents = request.POST['Dependents']
        Education = request.POST['Education']
        Self_Employed = request.POST['Self_Employed']
        ApplicantIncome = float(request.POST['ApplicantIncome'])
        CoapplicantIncome = float(request.POST['CoapplicantIncome'])
        LoanAmount = float(request.POST['LoanAmount'])
        Loan_Amount_Term = float(request.POST['Loan_Amount_Term'])
        Credit_History = float(request.POST['Credit_History'])
        Property_Area = request.POST['Property_Area']
        
        data = [[Gender, Married, Dependents, 
              Education, Self_Employed, ApplicantIncome,
              CoapplicantIncome, LoanAmount, Loan_Amount_Term,
              Credit_History, Property_Area]]
        
        df = pd.DataFrame(data, columns=columns)
        
        for col, encoder in encoders.items():
            df[col] = encoder.transform(df[col])

        pred = model.predict(df)
        if pred == 1:
            result = "Loan Approved"
        else:
            result = "Load Rejected"
        return render(request, "home.html", {"result":result})
        
    return render(request, "home.html")