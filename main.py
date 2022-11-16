from code.data import create_train_datasets, create_test_dataset
from code.models import create_model_and_optimizer, train_models
from code.models import setSeed, predict
from code.utils import train_stats_ploter

# Parameters 
POSITIVE_EXAMPLES="georges.csv"
NEGATIVE_EXAMPLES="non_georges.csv"

VALIDATION_SPLIT_P=0.1
NUM_WORKERS=2
LEARNING_RATE=0.001
MOMENTUM = 0.9
EPOCHS=20

TRAIN_FOLDER='train'
TEST_FOLDER='test/'

def calculate_test_performance(model):
    _, dataloader = create_test_dataset(TEST_FOLDER, num_workers=NUM_WORKERS)
    predict(model, dataloader, model_name='modvgg16.pt')

def train_model():
    datasets, dataloaders = create_train_datasets(POSITIVE_EXAMPLES, NEGATIVE_EXAMPLES, split=VALIDATION_SPLIT_P, num_workers=NUM_WORKERS, data_folder=TRAIN_FOLDER)
    model, optim, scheduler = create_model_and_optimizer('VGG16', lr=LEARNING_RATE, momentum=MOMENTUM)
    train_stats = train_models(model, optim, dataloaders, 
                              {ph:len(datasets[ph]) for ph in ['train', 'val']}, 
                              scheduler, epochs=EPOCHS, model_name='modvgg16.pt')
    train_stats_ploter(train_stats)

    del dataloaders
    del datasets

    calculate_test_performance(model)

def main():
    # Seed required for reproductivity
    setSeed(123456)

    train_model();

    # predict_with_model();

if __name__ == '__main__':
    main()