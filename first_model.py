import numpy as np
import pandas as pd


raw_df = pd.read_csv("/home/terrence/CODING/Python/MODELS/Credit_Union_PDs/default_data.csv", encoding="latin-1")
#file_name = "/home/terrence/CODING/Python/MODELS/CREDIT_UNION_PDS/default_data.csv"
#raw_df = pd.read_excel(file_name, sheet_name='Data')
print(raw_df.shape)

raw_df.dropna(inplace = True)

print(raw_df.shape)

'''
print(raw_df1.head(3))

# sample for speed
raw_df2 = raw_df.sample(frac=0.5,  replace=False)
print(raw_df2.shape)


# grab review text
#raw = list(raw_df2['Text'])
raw = raw_df2['Text'] #TERRENCE
raw = np.array(raw)
#print(raw)
print(len(raw))

'''










