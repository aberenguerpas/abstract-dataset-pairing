import subprocess
import requests
import argparse

def checkCall():
    try: 
        response = requests.post('http://localhost:5000/getEmbeddings', json = {'data':'a'})
        if response.status_code == 200:
            return True
        else:
            return False
    except Exception:
        return False

def classify(m):
    print("Getting classification with:", m)
    with subprocess.Popen(['./env/bin/python', 'classify.py', '-m', m, '-i','./data3/']) as proc:
        proc.communicate()

def index(m):
    print("Indexing with:", m)
    with subprocess.Popen(['./env/bin/python', 'index.py', '-m', m, '-i','./data3/']) as proc:
        proc.communicate()

def main():

    parser = argparse.ArgumentParser(description='Process WikiTables corpus')
    parser.add_argument('-t', '--type', default='all', choices=['all', 'index','clas'])

    args = parser.parse_args()
    models = ['rbt','sci','w2v','blo','stb','brt','fst']

    try:
        for m in models:
            with subprocess.Popen(['./env/bin/python', 'embeddings/main.py', '-m', m]) as proc:
                while proc.poll() is None:
                    if checkCall() == True:
                        if args.type == 'index' or args.type=="all":
                            index(m)
                        if args.type == 'clas' or args.type=="all":
                            classify(m)
                        proc.kill()

    except subprocess.CalledProcessError as err:
        print("Error occurred when running pgrep : {}".format(err))


if __name__ == "__main__":
    main()
