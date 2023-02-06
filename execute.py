import subprocess
import requests

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
    with subprocess.Popen(['./env/bin/python', 'classify.py', '-m', m, '-i','/raid/wake/data/']) as proc:
        proc.communicate()

def index(m):
    print("Indexing with:", m)
    with subprocess.Popen(['./env/bin/python', 'index.py', '-m', m, '-i','/raid/wake/data/']) as proc:
        proc.communicate()

def main():
    models = ['stb','brt', 'rbt', 'fst', 'w2v', 'sci', 'blo']

    try:
        for m in models:
            with subprocess.Popen(['./env/bin/python', 'embeddings/main.py', '-m', m]) as proc:
                while proc.poll() is None:
                    if checkCall() == True:
                        index(m)
                        classify(m)
                        proc.kill()

    except subprocess.CalledProcessError as err:
        print("Error occurred when running pgrep : {}".format(err))


if __name__ == "__main__":
    main()