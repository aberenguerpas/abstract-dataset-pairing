import pandas as pd
import os, json
from tqdm import tqdm
import numpy as np

def main():
    pd.set_option('display.float_format', lambda x: '%.3f' % x)
    files_path = './data3/'
    files = os.listdir(files_path)
    files = [i for i in files if i.endswith(".json")]

    results = []
    ignorados = 0

    for file in tqdm(files):
        try:
            dataset = dict()
            with open(files_path+file, 'r') as f:
                data = json.load(f)
                desc = data['abstract']
                n_words = len(desc.split(' '))
                dataset['n_words_abstract'] = n_words

                dataset['n_rows'] = []
                dataset['n_cols'] = []
                dataset['numeric_cols'] = []
                dataset['cat_cols'] = []

                for f in data['data']:
                    df = pd.read_csv('./data2/'+f, encoding = "ISO-8859-1", on_bad_lines='skip', engine='python', sep = None)

                    dataset['n_rows'].append(len(df.index))
                    dataset['n_cols'].append(len(df.columns))

                    if len(df.columns)>30 or len(df.index) >70:
                        print(f)

                    num_cols = 0
                    for col in df.columns:
                        if df[col].dtype.kind in 'biufc':
                            num_cols+=1
                    
                    dataset['numeric_cols'].append(num_cols)
                    dataset['cat_cols'].append(len(df.columns) - num_cols)

                dataset['n_rows'] = np.sum(dataset['n_rows'], axis=0)/len(data['data'])
                dataset['n_cols'] = np.sum(dataset['n_cols'], axis=0)/len(data['data'])
                dataset['numeric_cols'] = np.sum(dataset['numeric_cols'], axis=0)/len(data['data'])
                dataset['cat_cols'] = np.sum(dataset['cat_cols'], axis=0)/len(data['data'])
                                   
            results.append(dataset)

        except Exception as e:
            print(e)
            ignorados+=1
            
    print('Ignored', ignorados)

    df = pd.DataFrame.from_dict(results)

    print(df.columns)

    print('N?? tables', len(df))
    print('N?? rows', df.loc[:,'n_rows'].sum())
    print('N?? cols', df.loc[:,'n_cols'].sum())

    print('N?? numeric cols', df.loc[:,'numeric_cols'].sum())
    print('N?? categorical cols', df.loc[:,'cat_cols'].sum())

    print(df.describe())
if __name__ == "__main__":
    main()