import pandas as pd

def main():
    df = pd.read_csv('./results_w2v.csv')
    print(df.mean(axis=0))


if __name__ == "__main__":
    main()