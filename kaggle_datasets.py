from kaggle.api.kaggle_api_extended import KaggleApi
import os, json, glob
import shutil
import traceback

api = KaggleApi()
api.authenticate()
count = 1

path = os.path.join(os.getcwd(),'data2')
data_list = [1]
while len(data_list)!=0:
   
    data_list = api.dataset_list(page=count, file_type='csv', tag_ids='Computer Science', max_size=20000000, min_size=10000, license_name='cc')
    count+=1
    for data in data_list:
        try:
            api.dataset_metadata(data.ref, path=path)
            
            api.dataset_download_cli(data.ref, path=os.path.join(path,'aux'), unzip=True, force=True, quiet=True)
            with open(os.path.join(path,'dataset-metadata.json'), 'r') as f:
                meta = json.load(f)
                root_directory = os.path.join(path,'aux')
                patron = '**/*.*'
                archivos = []
                for ruta, directorios, nombres_archivos in os.walk(root_directory):
                    archivos.extend(glob.glob(os.path.join(ruta, patron), recursive=True))

                meta['data'] = [str(meta['id_no'])+a.split("/")[-1:][0] for a in archivos]
                
            with open(os.path.join(path,'dataset-metadata.json'), 'w') as json_file:
                json.dump(meta, json_file)

            root_directory = os.path.join(path,'aux')
            patron = '**/*.*'
            for ruta, directorios, nombres_archivos in os.walk(root_directory):
                for f in glob.glob(os.path.join(ruta, patron), recursive=True):
                    shutil.move(os.path.join(ruta,f), path)
                  
                    os.rename(os.path.join(path,f.split("/")[-1:][0]), os.path.join(path, str(meta['id_no'])+f.split(".")[0].split("/")[-1:][0]+".csv"))

            os.rename(os.path.join(path,'dataset-metadata.json'), os.path.join(path, data.ref.replace("/","-") +'.json'))
           
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            pass
        
    print(20*count)
 

