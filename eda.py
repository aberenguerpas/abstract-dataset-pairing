import pandas as pd
import os, json
from tqdm import tqdm
import numpy as np

def main():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    files_path = '/raid/wake/data/'
    files = os.listdir(files_path)
    files = [i for i in files if i.endswith(".json")]

    results = []
    ignorados = 0

    for file in tqdm(files):
        try:
            dataset = dict()
            with open(files_path+file, 'r') as f:
                data = json.load(f)
                desc = data['desc'].replace('&nbsp;',' ')
                n_words = len(desc.split(' '))
                dataset['n_words_abstract'] = n_words

                dataset['n_rows'] = []
                dataset['n_cols'] = []
                dataset['numeric_cols'] = []
                dataset['cat_cols'] = []

                for f in data['files']:
                    df = pd.read_csv(files_path+f, encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None)

                    dataset['n_rows'].append(len(df.index))
                    dataset['n_cols'].append(len(df.columns))

                    num_cols = 0
                    for col in df.columns:
                        if df[col].dtype.kind in 'biufc':
                            num_cols+=1
                    
                    dataset['numeric_cols'].append(num_cols)
                    dataset['cat_cols'].append(len(df.columns) - num_cols)


                dataset['n_rows'] = np.mean(dataset['n_rows'], axis=0)
                dataset['n_cols'] = np.mean(dataset['n_cols'], axis=0)
                dataset['numeric_cols'] = np.mean(dataset['numeric_cols'], axis=0)
                dataset['cat_cols'] = np.mean(dataset['cat_cols'], axis=0)
                                   
                results.append(dataset)

            df2 = pd.DataFrame.from_dict(results)
            print(df2)
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

    print(df)
    print(df.describe())
if __name__ == "__main__":
    main()