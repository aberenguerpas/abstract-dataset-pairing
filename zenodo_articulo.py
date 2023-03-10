# Dowload experiment data

import requests
import re, json
from collections import defaultdict
import time

def download(file_name, url):
  r = requests.get(url, allow_redirects=True)
  file_n = '/raid/wake/data/'+file_name
  open(file_n.replace("'",""), 'wb').write(r.content)

page = 1
obtained = 0
total = 1

while obtained < total:
  response = requests.get('https://zenodo.org/api/records',
                          params={'type': 'publication','file_type':'csv','subtype':'article', 'size':100, 'page':page, 'access_token':'bVwiOHaEAPuDZHqlxBdnCjQB2usbJpQuZRwdNZ6y08JDQlrKdNgtkRGQlMwR'})
  total =  response.json()['hits']['total']

  obtained += len(response.json()['hits']['hits'])
  print(obtained,'/',total)
  data = []
  for d in response.json()['hits']['hits']:
    try:
      r = requests.get('https://zenodo.org/api/records/'+str(d['id']))

      doc = defaultdict()

      desc = re.sub('<[^<]+?>', '', r.json()['metadata']['description'])

      if len(desc.split(" ")) > 100:
        doc['id'] = d['id']
        doc['desc'] = desc
        doc['files'] = []

        for file in r.json()['files']:
          if file['type']=='csv':
            file_name = file['key']
            doc['files'].append(str(doc['id'])+file_name)

            download(str(doc['id'])+file_name, file['links']['self'])
            with open('/raid/wake/data/'+str(doc['id'])+".json", "w") as outfile:
              json.dump(doc, outfile)
    except Exception as e:
      print(e)
      time.sleep(60)
  page+=1