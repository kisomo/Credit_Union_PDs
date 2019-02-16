import numpy as np
import pandas as pd


raw_df = pd.read_csv("/home/terrence/CODING/Python/MODELS/Credit_Union_PDs/default_data.csv", encoding="latin-1")
#file_name = "/home/terrence/CODING/Python/MODELS/CREDIT_UNION_PDS/default_data.csv"
#raw_df = pd.read_excel(file_name, sheet_name='Data')
print(raw_df.shape)
#raw_df.dropna(inplace = True)
#print(raw_df.shape)
#print(raw_df.columns.values)

'''

[u'Loan Number' u'Loan Type Description' u'Balance' u'Loan Term' u'Interest Rate' u'Origination Date' u'Origination Month'
 u'Most Recent Credit Score' u'AmountFunded' u'MonthlyIncomeBaseSalary' u'TotalMonthlyIncome' u'MonthlyIncomeOther'
 u'Collateral Current Valuation' u'LTV' u'Number of Days Delinquent' u'Days Late T or F' u'Balance.1' u'Days 11-15 Delinquent'
 u'Days 16-20 Delinquent' u'Days 21-29 Delinquent' u'Days 30-44 Delinquent' u'Days 45-59 Delinquent' u'Days 60-179 Delinquent'
 u'Days 180-359 Days Delinquent' u'Days 360+ Delinquent' u'Days Delinquent T or F' u'Grade Overall' u'Original Loan Amount'
 u'Current Credit Limit' u'Maturity Date' u'Maturity Month' u'Original Credit Score' u'LTV-Original' u'Probability of Default'
 u'Branch' u'Loan Officer' u'Underwriter' u'Loan Type Code' u'Loan Category' u'Auto Dealer' u'Primary Customer City' u'Status'
 u'Updated Credit Score' u'Original Interest Rate' u'LTV (Effective)' u'LTV-Original (Effective)' u'LTV-Original Total Commitments'
 u'LTV-Total Commitments' u'LTV-Total Commitments (Effective)' u'LTV-Total Commitments-Original (Effective)'
 u'Grade by Most Recent Credit Score' u'Grade by Cerdit Score (ORIGINAL)' u'GRADE BY CREDIT SCORE (UPDATED)' u'JointTotalMonthlyIncome'
 u'JointProfessionMonths' u'JointCity' u'JointApplicantType' u'JointMonthlyIncomeBaseSalary' u'JointMonthlyIncomeOther'
 u'JointMonthlyIncomeOtherDescription1' u'JointOccupation' u'IndCity' u'IndMonthlyIncomeBaseSalary' u'IndMonthlyIncomeOther'
 u'IndTotalMonthlyIncome' u'IndMonthlyIncomeOtherDescription1' u'PaymentAmount' u'PaymentFrequency' u'Insurance' u'DueDay1' u'DueDay2'
 u'PaymentMethodText' u'SymitarPurposeCode' u'ApprovedLTV' u'FundedLTV' u'PaymentToIncome' u'NumberOfOpenRevolvingAccounts' u'AmountApproved'
 u'AmountFunded.1' u'AmountOwedToLender' u'DOB' u'DOB.1' u'DOB.2' u'AGE' u'AGE of BORROWER' u'JointDOB' u'Year' u'Year.1' u'AGE OF JOINT'
 u'AGE OF JOINT.1' u'IndDOB' u'YEAR' u'YEAR.1' u'AGE.1' u'AGE of IND' u'AllButThisDebtToIncomeFund' u'AllButThisDebtToIncomeUW'
 u'EstimatedMonthlyPayment' u'TotalDebtToIncomeFund' u'TotalDebtToIncomeUW' u'TotalUnsecureBalance' u'TotalExistingLoanAmount' u'APR'
 u'IsHighRiskConsumerLoan' u'IsAdvanceRequest' u'IsWorkoutLoan' u'LoanPaymentFrequency' u'PaymentType' u'Rate']

'''

raw_df['label'] = raw_df['Number of Days Delinquent'].map(lambda x : 1 if int(x) > 11 else 0)

print(raw_df.shape)

#print(raw_df['Loan Type Description'].mean())
print(np.any(np.isnan(raw_df['Loan Type Description'])))
#print(raw_df['Balance'].mean())
print(np.any(np.isnan(raw_df['Balance'])))
#print(raw_df['Loan Term'].mean())
print(np.any(np.isnan(raw_df['Loan Term'])))
#print(raw_df['LTV'].mean())
print(np.any(np.isnan(raw_df['LTV'])))
#print(raw_df['label'].sum())
print(np.any(np.isnan(raw_df['label'])))

print("\n\n")

#print(raw_df['Interest Rate'].mean())
print(np.any(np.isnan(raw_df['Interest Rate'])))
#print(raw_df['Origination Month'].mean())
print(np.any(np.isnan(raw_df['Origination Month'])))
#print(raw_df['Most Recent Credit Score'].mean())
print(np.any(np.isnan(raw_df['Most Recent Credit Score'])))
#print(raw_df['AmountFunded'].mean())
raw_df['AmountFunded'] = raw_df['AmountFunded'].fillna(-99999)
print(np.any(np.isnan(raw_df['AmountFunded'])))
#print(raw_df['MonthlyIncomeBaseSalary'].mean())
raw_df['MonthlyIncomeBaseSalary'] = raw_df['MonthlyIncomeBaseSalary'].fillna(-99999)
print(np.any(np.isnan(raw_df['MonthlyIncomeBaseSalary'])))
#print(raw_df['TotalMonthlyIncome'].mean())
raw_df['TotalMonthlyIncome'] = raw_df['TotalMonthlyIncome'].fillna(-99999)
print(np.any(np.isnan(raw_df['TotalMonthlyIncome'])))
#print(raw_df['MonthlyIncomeOther'].mean())
raw_df['MonthlyIncomeOther'] = raw_df['MonthlyIncomeOther'].fillna(-99999)
print(np.any(np.isnan(raw_df['MonthlyIncomeOther'])))
#print(raw_df['Collateral Current Valuation'].mean())
print(np.any(np.isnan(raw_df['Collateral Current Valuation'])))


#raw_df.fillna(-99999)

#df1 = pd.concat([raw_df['Loan Type Description'], raw_df['Balance'], raw_df['Loan Term'],raw_df['LTV'], raw_df['label']],axis =1)
#print(df1.shape)

df1 = pd.concat([raw_df['Loan Type Description'], raw_df['Balance'], raw_df['Loan Term'],raw_df['Interest Rate'],
raw_df['Origination Month'],raw_df['Most Recent Credit Score'],
raw_df['AmountFunded'],raw_df['MonthlyIncomeBaseSalary'],raw_df['TotalMonthlyIncome'],raw_df['MonthlyIncomeOther'],
raw_df['Collateral Current Valuation'],raw_df['LTV'], raw_df['label']],axis = 1)
print(df1.shape)

print(df1.head(4))

#df1 = df1.reset_index()

print(np.any(np.isnan(df1)))
print(np.all(np.isfinite(df1)))

y_CU = raw_df['Probability of Default']
y = df1.label
X = df1.drop("label", axis =1)

print(X.shape)

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, auc, roc_curve


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr = lr.fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
confusion = confusion_matrix(y_test, lr_predicted)
print(lr.score(X_test,y_test))
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != lr_predicted).sum()))

print("\n\n")

print(confusion)

y_pred = lr.predict(X_test)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc = auc(fpr,tpr)

print("\n\n")

print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0],(y_test != y_pred).sum()))

print("\n\n")

print("Logistic accuracy:" ,acc)
print("Logistic precision:" ,prec)
print("Logistic recall:" ,rec)
print("Logistic f1 ratio:" ,f1)
print("Logistic AUC:" ,auc)

y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
print(y_proba_lr[:,1])
#print(y_CU - y_proba_lr[:,1])
#print(y_CU)

#y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))
# show the probability of positive class for first 20 instances
#print(y_proba_list)












