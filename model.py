import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

Personal_Loan = pd.read_csv("Bank_Personal_Loan_Modelling.csv")

x = Personal_Loan[['Age','Experience','Income','Family']]
y = Personal_Loan['Personal Loan']

# Initiatlize the model
logreg = LogisticRegression(solver='liblinear', random_state = 0)

# Fit the model
logreg.fit(x, y)  

pickle.dump(logreg, open('loan_model.pkl', 'wb'))  