import os
import pandas as pd

def isNum(df):
    num_cols = 0
    for col in df.columns:
        col = col.replace('-','')
        col = col.replace('.','')
        if col.isnumeric():
            num_cols+=1
    return num_cols/len(df.columns)

files_path = 'data/'
files = os.listdir(files_path)

files = [i for i in files if i.endswith(".csv")]
files.sort()


for file in files:
            # Headers
            df = pd.read_csv(files_path+file , encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None, nrows=1020)
            if isNum(df)>0.2:
                print(file)