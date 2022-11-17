from code.data import create_train_datasets, create_test_dataset
from code.models import create_model_and_optimizer, train_models
from code.models import setSeed, predict, predict_single_image
from code.utils import train_stats_ploter
from code.parameters import check_params
import sys

# Parameters 
PROGRAM_PARAMS = {'image':"", "notrain":False}
POSITIVE_EXAMPLES="georges.csv"
NEGATIVE_EXAMPLES="non_georges.csv"

VALIDATION_SPLIT_P=0.1
NUM_WORKERS=2
LEARNING_RATE=0.0001 #0.001
MOMENTUM = 0.9
EPOCHS=20
BACTH_SIZE=32

TRAIN_FOLDER='train'
TEST_FOLDER='test/'

MODEL_NAME='modresnet.pt' #'modvgg16.pt'
MODEL_TYPE='RESNET'

# BEST LR 'VGG16': 0.001 Adagrad
# BEST LR 'RESNET': 0.0001 Adam

def calculate_test_performance(model):
    _, dataloader = create_test_dataset(TEST_FOLDER, batch_size=BACTH_SIZE, num_workers=NUM_WORKERS)
    predict(model, dataloader, model_name=MODEL_NAME)

def train_model():
    '''
        Train the model used to detect st. george
    '''
    datasets, dataloaders = create_train_datasets(POSITIVE_EXAMPLES, NEGATIVE_EXAMPLES, batch_size=BACTH_SIZE, split=VALIDATION_SPLIT_P, num_workers=NUM_WORKERS, data_folder=TRAIN_FOLDER)
    model, optim, scheduler = create_model_and_optimizer(MODEL_TYPE, lr=LEARNING_RATE, momentum=MOMENTUM)
    train_stats = train_models(model, optim, dataloaders, 
                              {ph:len(datasets[ph]) for ph in ['train', 'val']}, 
                              scheduler, epochs=EPOCHS, model_name=MODEL_NAME)
    train_stats_ploter(train_stats)

    del dataloaders
    del datasets

    calculate_test_performance(model)

def predict_single(image_path:str):
    '''
        Detect if in the image st. geroge is present

        Parameters:
            image_path:str The path to the image to classify.
    '''
    model, _, _ = create_model_and_optimizer(MODEL_TYPE)
    predict_single_image(model, image_path, model_name=MODEL_NAME)

def main():
    if check_params(arg=sys.argv[1:], params=PROGRAM_PARAMS) == 0:
        exit(0)

    # Seed required for reproductivity
    setSeed(123456)

    if not PROGRAM_PARAMS['notrain']:
        train_model();
    
    if len(PROGRAM_PARAMS['image']) > 0:
        predict_single(PROGRAM_PARAMS['image']);

if __name__ == '__main__':
    main()