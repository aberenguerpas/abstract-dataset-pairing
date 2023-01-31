import sys
import subprocess
import time
import requests

models = ['stb', 'brt', 'rbt', 'fst', 'w2v', 'sci', 'blo']


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
    print("Obteniendo clasificacion con modelo:", m)
    with subprocess.Popen(['./env/bin/python', 'classify.py', '-m', m, '-i','data/'], 
                            stdout=subprocess.PIPE,
                            text=True,
                            stderr=subprocess.STDOUT) as proc:
        while True:
            output = proc.stdout.readline()
            if proc.poll() is not None:
                break
            if output:
                print(output)

def index(m):
    print("Indexando con modelo:", m)
    with subprocess.Popen(['./env/bin/python', 'index.py', '-m', m, '-i','data/'], 
                            stdout=subprocess.PIPE,
                            text=True,
                            bufsize=0,
                            stderr=subprocess.STDOUT) as proc:
        while True:
            output = proc.stdout.readline()
            if output=='' or proc.poll() is not None:
                break
            if output:
                print(output.strip())

def main():
    try:
        for m in models:
            with subprocess.Popen(['./env/bin/python','-u', 'embeddings/main.py', '-m', m], 
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE) as proc:
                while proc.poll() is None:
                    if checkCall() == True:
                        index(m)
                        classify(m)
                        proc.kill()

    except subprocess.CalledProcessError as err:
        print("error occurred when running pgrep : {}".format(err))


if __name__ == "__main__":
    main()