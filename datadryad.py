from tqdm import tqdm
import requests
import traceback
import json
from datetime import timedelta
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=29, period=timedelta(seconds=60).total_seconds())
def get_file(id):
    URL = 'https://datadryad.org/api/v2/files/'+str(id)
    r = requests.get(url = URL)
    try:
        if r.status_code != 204:
            data = r.json()

            if data['size']>20000000 or data['mimeType'] != 'text/csv':
                return None
            print(data)
            file = requests.get('https://datadryad.org/api/v2/files/'+str(id)+'/download', allow_redirects=True)

            open("./data/"+data['path'], 'wb').write(file.content)

            return data['path']
    except Exception as e:
        print(traceback.format_exc())
        print(e)

@sleep_and_retry
@limits(calls=1, period=timedelta(seconds=60).total_seconds())
def search(i):
    URL = 'https://datadryad.org/api/v2/datasets?per_page=100&page='+str(i)
    r = requests.get(url = URL)

    return r.json()

count = 1
total = 519

for i in tqdm(range(519), desc="Total"):

    data = search(i)

    for element in tqdm(data['_embedded']['stash:datasets'], desc="partial"):
        meta = dict()
        meta['id_no'] = element['id']
        meta['doi'] = element['identifier']
        meta['abstract'] = element['abstract']
        meta['data'] = get_file(element['id'])

        if meta['data'] is not None:
            with open("./data/"+str(element['id'])+".json", "w") as outfile:
                json.dump(meta, outfile)
