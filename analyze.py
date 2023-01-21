import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description='Analyze results')
    parser.add_argument('-i', '--input', default='results/', help='Name of the input folder storing CSV tables')

    args = parser.parse_args()
    df = pd.read_csv(args.input)

    print(df.mean(axis=0))

if __name__ == "__main__":
    main()