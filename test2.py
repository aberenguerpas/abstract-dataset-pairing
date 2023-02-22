import pandas as pd
import json
from bs4 import BeautifulSoup as bs
from html import escape
import pandas as pd
import warnings


def format_html(img):
    ''' Formats HTML code from tokenized annotation of img
    '''
    html_code = img['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], img['html']['cells'][::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = '''<html>
                   <head>
                   <meta charset="UTF-8">
                   </head>
                   <body>
                   <table frame="hsides" rules="groups" width="100%%">
                     %s
                   </table>
                   </body>
                   </html>''' % html_code

    # prettify the html
    soup = bs(html_code,features="html5lib")
    html_code = soup.prettify()
    return html_code

warnings.simplefilter(action='ignore', category=FutureWarning)

with open('/raid/wake/raw_data/pubtabnet/PubTabNet_2.0.0.jsonl', "rb") as f:
    for line in f:
        try:
            data = json.loads(line)
            table_MN = pd.read_html(format_html(data), flavor='html5lib')

            table_MN[0].to_csv("./data2/"+data['filename'][:-4]+'.csv', index=False)
        except Exception as e:
            print(e)
            
        # table_MN.columns = table_MN.columns.to_flat_index()