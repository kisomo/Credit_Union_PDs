import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt



#raw_df = pd.read_csv("/home/terrence/CODING/Python/MODELS/Credit_Union_PDs/default_data.csv", encoding="latin-1")
myfile = "/home/terrence/CODING/Python/MODELS/Credit_Union_PDs/Test Variables READY.xlsx"
raw_df = pd.read_excel(myfile, sheetname = 'Data', header = 0)
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
raw_df['AmountFunded'] = raw_df['AmountFunded'].fillna(raw_df['AmountFunded'].mean())
print(np.any(np.isnan(raw_df['AmountFunded'])))
#print(raw_df['MonthlyIncomeBaseSalary'].mean())
raw_df['MonthlyIncomeBaseSalary'] = raw_df['MonthlyIncomeBaseSalary'].fillna(raw_df['MonthlyIncomeBaseSalary'].mean())
print(np.any(np.isnan(raw_df['MonthlyIncomeBaseSalary'])))
#print(raw_df['TotalMonthlyIncome'].mean())
raw_df['TotalMonthlyIncome'] = raw_df['TotalMonthlyIncome'].fillna(raw_df['TotalMonthlyIncome'].mean())
print(np.any(np.isnan(raw_df['TotalMonthlyIncome'])))
#print(raw_df['MonthlyIncomeOther'].mean())
raw_df['MonthlyIncomeOther'] = raw_df['MonthlyIncomeOther'].fillna(raw_df['MonthlyIncomeOther'].mean())
print(np.any(np.isnan(raw_df['MonthlyIncomeOther'])))
#print(raw_df['Collateral Current Valuation'].mean())
print(np.any(np.isnan(raw_df['Collateral Current Valuation'])))
print("\n\n")
#raw_df['Balance'] = raw_df['Balance'].fillna(-99999)
print(np.any(np.isnan(raw_df['Balance'])))
#raw_df['Grade Overall'] = raw_df['Grade Overall'].fillna(-99999)
print(np.any(np.isnan(raw_df['Grade Overall'])))
#raw_df['Current Credit Limit'] = raw_df['Current Credit Limit'].fillna(-99999)
print(np.any(np.isnan(raw_df['Current Credit Limit'])))
#raw_df['Loan Type Code'] = raw_df['Loan Type Code'].fillna(-99999)
print(np.any(np.isnan(raw_df['Loan Type Code'])))
#raw_df['Status'] = raw_df['Status'].fillna(-99999)
print(np.any(np.isnan(raw_df['Status'])))
raw_df['Insurance'] = raw_df['Insurance'].fillna(raw_df['Insurance'].mean())
print(np.any(np.isnan(raw_df['Insurance'])))
raw_df['NumberOfOpenRevolvingAccounts'] = raw_df['NumberOfOpenRevolvingAccounts'].fillna(raw_df['NumberOfOpenRevolvingAccounts'].mean())
print(np.any(np.isnan(raw_df['NumberOfOpenRevolvingAccounts'])))
raw_df['APR'] = raw_df['APR'].fillna(raw_df['APR'].mean())
print(np.any(np.isnan(raw_df['APR'])))

#raw_df['PaymentToIncome'] = raw_df['PaymentToIncome'].fillna(raw_df['PaymentToIncome'].mean())
#print(np.any(np.isnan(raw_df['PaymentToIncome'])))

raw_df['AmountOwedToLender'] = raw_df['AmountOwedToLender'].fillna(raw_df['AmountOwedToLender'].mean())
print(np.any(np.isnan(raw_df['AmountOwedToLender'])))

#raw_df['AGE of BORROWER'] = raw_df['AGE of BORROWER'].fillna(raw_df['AGE of BORROWER'].mean())
#print(np.any(np.isnan(raw_df['AGE of BORROWER'])))

raw_df['LoanPaymentFrequency'] = raw_df['LoanPaymentFrequency'].fillna(raw_df['LoanPaymentFrequency'].mean())
print(np.any(np.isnan(raw_df['LoanPaymentFrequency'])))

raw_df['Rate'] = raw_df['Rate'].fillna(raw_df['Rate'].mean())
print(np.any(np.isnan(raw_df['Rate'])))

#df1 = pd.concat([raw_df['Loan Type Description'], raw_df['Balance'], raw_df['Loan Term'],raw_df['LTV'], raw_df['label']],axis =1)

df1 = raw_df[['Loan Type Description','Balance','Loan Term','Interest Rate','Origination Month','Most Recent Credit Score',
'AmountFunded','MonthlyIncomeBaseSalary', 'TotalMonthlyIncome','MonthlyIncomeOther','Collateral Current Valuation','LTV', 
'Balance','Grade Overall','Current Credit Limit','Loan Type Code','Loan Category','Status','Updated Credit Score',
'Original Interest Rate','Grade by Cerdit Score (ORIGINAL)','GRADE BY CREDIT SCORE (UPDATED)','Insurance',
'NumberOfOpenRevolvingAccounts','AmountOwedToLender','APR','LoanPaymentFrequency','Rate','label']]

print(df1.shape)

print(df1.head(4))


#df1 = df1.reset_index()
#print(np.any(np.isnan(df1)))
#print(np.all(np.isfinite(df1)))

y_CU = raw_df['Probability of Default']
y = df1.label
X = df1.drop("label", axis =1)

print(X.shape)

RANDOM_SEED = 42
LABELS = ["non-delinguent", "delinguent"]

print(df1.shape)
print(df1.isnull().values.any())
print(df1.head(3))

count_classes = pd.value_counts(df1['label'], sort = True)
count_classes.plot(kind = 'bar', rot=0)
plt.title("delinguency distribution")
plt.xticks(range(2), LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()


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

#y_proba_lr = lr.fit(X_train, y_train).predict_proba(X_test)
y_proba_lr = lr.fit(X_train, y_train).predict_proba(X)
print(y_proba_lr[:,1])
#print(y_CU - y_proba_lr[:,1])
#print(y_CU)

#y_proba_list = list(zip(y_test[0:20], y_proba_lr[0:20,1]))
# show the probability of positive class for first 20 instances
#print(y_proba_list)


from sklearn.model_selection import cross_val_score

# accuracy is the default scoring metric
print('Cross-validation (accuracy)', cross_val_score(lr, X, y, cv=5))
# use AUC as scoring metric
print('Cross-validation (AUC)', cross_val_score(lr, X, y, cv=5, scoring = 'roc_auc'))
# use recall as scoring metric
print('Cross-validation (recall)', cross_val_score(lr, X, y, cv=5, scoring = 'recall'))
print('Cross-validation (precision)', cross_val_score(lr, X, y, cv=5, scoring = 'precision'))


import seaborn as sns   

#cm = pd.crosstab(y_test, y_pred, rownames = 'True', colnames = 'predicted', margins = False)
cm = confusion_matrix(y_test, lr_predicted)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['non-deli', 'deli']); ax.yaxis.set_ticklabels(['non-deli', 'deli'])
plt.show()


y_scores_lr = lr.decision_function(X_test)

# ### Precision-recall curves

from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
closest_zero = np.argmin(np.abs(thresholds))
closest_zero_p = precision[closest_zero]
closest_zero_r = recall[closest_zero]

plt.figure()
plt.xlim([0.0, 1.01])
plt.ylim([0.0, 1.01])
plt.plot(precision, recall, label='Precision-Recall Curve')
plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
plt.xlabel('Precision', fontsize=16)
plt.ylabel('Recall', fontsize=16)
plt.axes().set_aspect('equal')
plt.show()



fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
roc_auc_lr = 72 #auc(fpr_lr, tpr_lr)

fig1 = plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fpr_lr, tpr_lr, lw=3, label='Logistic Reg ROC curve (area = {:0.2f})'.format(roc_auc_lr))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('ROC curve (delinguency classifier)', fontsize=16)
plt.legend(loc='lower right', fontsize=13)
plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
plt.axes().set_aspect('equal')
plt.show()
fig1.savefig("ROC_curve_1.pdf")


print(y_proba_lr[:,1])
err = y_CU - y_proba_lr[:,1]
rmse_err = np.sqrt(np.mean(err**2))
print(rmse_err)


#https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py


from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
# Binarize the output
y_2 = y #label_binarize(y, classes=[0, 1])
print(y_2.shape)
n_classes = 1 # y_2.shape[1]

random_state = np.random.RandomState(0)

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_2, test_size=.25, random_state=0)

# Learn to predict each class against the other
#classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=random_state))
#y_score = classifier.fit(X_train, y_train).decision_function(X_test)
lr = LogisticRegression()
y_score = lr.fit(X_train, y_train).decision_function(X_test)


# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = 72 #auc(fpr, tpr)

fig =plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
fig.savefig("ROC_curve_2.pdf")









