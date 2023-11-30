import argparse
import pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    with open(args.filename, 'rb') as f:
        data = pickle.load(f)
    print(data)

if __name__ == '__main__':
    main()