from code.data import create_train_datasets

POSITIVE_EXAMPLES="georges.csv"
NEGATIVE_EXAMPLES="non_georges.csv"
VALIDATION_SPLIT_P=0.1
NUM_WORKERS=4

def train_model():
    datasets, dataloaders = create_train_datasets(POSITIVE_EXAMPLES, NEGATIVE_EXAMPLES, split=VALIDATION_SPLIT_P, num_workers=NUM_WORKERS)


def main():
    train_model();

if __name__ == '__main__':
    main()