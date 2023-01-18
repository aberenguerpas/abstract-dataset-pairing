import pandas as pd
import os, json
from tqdm import tqdm
import numpy as np

def main():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    files_path = './data/'
    files = os.listdir(files_path)

    results = []
    ignorados = 0

    for file in tqdm(files):
        try:
            if file[-4:] == 'json':
                dataset = dict()
                with open(files_path+file, 'r') as f:
                    data = json.load(f)
                    desc = data['desc'].replace('&nbsp;',' ')
                    n_words = len(desc.split(' '))
                    dataset['n_words_abstract'] = n_words

                    df = pd.read_csv(files_path+str(data['id'])+'.csv', encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None)

                    dataset['n_rows'] = len(df.index)
                    dataset['n_cols'] = len(df.columns)

                    num_cols = 0
                    for col in df.columns:
                        if df[col].dtype.kind in 'biufc':
                            num_cols+=1

                    dataset['numeric_cols'] = num_cols
                    dataset['cat_cols'] = len(df.columns) - dataset['numeric_cols']

                    print(file, dataset['numeric_cols'], dataset['cat_cols'])
                    
                results.append(dataset)
        except Exception as e:
            print(e)
            ignorados+=1
            
    print('Ignored', ignorados)

    df = pd.DataFrame.from_dict(results)

    print(df.columns)

    print('Nº tables', len(df))
    print('Nº rows', df.loc[:,'n_rows'].sum())
    print('Nº cols', df.loc[:,'n_cols'].sum())

    print('Nº numeric cols', df.loc[:,'numeric_cols'].sum())
    print('Nº categorical cols', df.loc[:,'cat_cols'].sum())

    print(df.describe())
if __name__ == "__main__":
    main()