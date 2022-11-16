from code.data import create_train_datasets
from code.models import create_model_and_optimizer, train_models
from code.utils import train_stats_ploter

POSITIVE_EXAMPLES="georges.csv"
NEGATIVE_EXAMPLES="non_georges.csv"
VALIDATION_SPLIT_P=0.1
NUM_WORKERS=4
LEARNING_RATE=0.001
MOMENTUM = 0.9
EPOCHS=20

def train_model():
    datasets, dataloaders = create_train_datasets(POSITIVE_EXAMPLES, NEGATIVE_EXAMPLES, split=VALIDATION_SPLIT_P, num_workers=NUM_WORKERS)
    model, optim, scheduler = create_model_and_optimizer('VGG16', lt=LEARNING_RATE, momentum=MOMENTUM)
    train_stats = train_models(model, optim, dataloaders, 
                              {ph:len(datasets[ph]) for ph in ['train', 'val']}, 
                              scheduler, epochs=EPOCHS, model_name='modvgg16.pt')
    train_stats_ploter(train_stats)

def main():
    train_model();

if __name__ == '__main__':
    main()