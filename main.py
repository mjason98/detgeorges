from code.utils import data_downloader

POSITIVE_EXAMPLES="data/georges.csv"
NEGATIVE_EXAMPLES="data/no_georges.csv"


def main():
    data_downloader(POSITIVE_EXAMPLES)

if __name__ == '__main__':
    main()