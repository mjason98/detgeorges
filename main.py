from code.utils import data_images_downloader

POSITIVE_EXAMPLES="data/georges.csv"
NEGATIVE_EXAMPLES="data/non_georges.csv"


def main():
    data_images_downloader(POSITIVE_EXAMPLES, image_folder="pos", data_folder="data")
    data_images_downloader(NEGATIVE_EXAMPLES, image_folder="neg", data_folder="data")

if __name__ == '__main__':
    main()