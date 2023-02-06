import os
import pandas as pd

files_path = '/raid/wake/data/'
files = os.listdir(files_path)

files = [i for i in files if i.endswith(".csv")]
files.sort()

for file in files:
            # Headers
            print(file)
            df = pd.read_csv(files_path+file , encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None, nrows=1020)
            print(df.columns.values)