import json
import sklearn.metrics as metrics
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os, time
import random
import numpy as np

from .data import get_test_data_transform
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setSeed(my_seed:int):
    torch.manual_seed(my_seed)
    np.random.seed(my_seed)
    random.seed(my_seed)

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))

def save_model(model, path):
    torch.save(model.state_dict(), path) 

def create_model_and_optimizer(model:str, lr=0.001, momentum=0.9):
    '''
    Creates models and optimizers

    Parameters:

        model:str the arquitecture of the model: ['VGG16', 'RESNET']

        lr: learning rate
    '''
    
    if model == "VGG16":
        model_ft = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # replace the number of classes in the last layer by 2
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier[-1] =  nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)

        # for now, a simple optimizer
        optimizer_ft = optim.Adagrad(model_ft.parameters(), lr=lr)#, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = optim.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        return model_ft, optimizer_ft, None#exp_lr_scheduler
    elif model == "RESNET":
        model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        num_ftrs = model_ft.fc.in_features
        model_ft.fc =  nn.Linear(num_ftrs, 2)
        model_ft = model_ft.to(device)

        # for now, a simple optimizer
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=lr)#, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = optim.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        return model_ft, optimizer_ft, None#exp_lr_scheduler
    else:
        raise Exception(f"Model {model} is not allowed")


def train_models(model, optimizer, dataloaders:dict, dataset_sizes:dict, scheduler=None, epochs=20, model_name='model.pt'):
    '''
    Train the model given the optimizer and the data provided.
    '''
    
    if not os.path.isdir("pts"):
        os.mkdir("pts")

    save_path = os.path.join("pts", model_name)
    
    since = time.time()
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()

    epoch_stats = {'train':[], 'val':[]}

    for epoch in range(epochs):
        # training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            dataset_size = dataset_sizes[phase]

            iter = tqdm(dataloaders[phase])
            
            for batch in iter:

                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double().item() / dataset_size
            
            epoch_stats[phase].append((epoch_loss, epoch_acc))

            print (f"{phase} epoch {epoch}/{epochs}: loss {epoch_loss} acc {epoch_acc}\n")
            
            #  best model
            if epoch_acc > best_acc and phase == 'val':
                best_acc = epoch_acc
                save_model(model, save_path)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return epoch_stats

def predict(model, dataloader, model_name='model.pt', load_saved_model=True, show_metrics=True):
    '''
        Calculate the metrics given a pre-trained model on the test data set.
    '''
    if not os.path.isdir('pts'):
        os.mkdir('pts')
    
    if load_saved_model:
        load_model(model, os.path.join('pts', model_name))
    
    cpu0 = torch.device("cpu")
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        iter = tqdm(dataloader)
        for batch in iter:

            inputs = batch[0].to(device)
            labels = batch[1].to(device).squeeze()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            y_true.append(labels.to(cpu0).numpy())
            y_pred.append(preds.squeeze().to(cpu0).numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # todo: optimize this calculations
    acc = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)

    if show_metrics:
        print (f"\n# Metrics:  acc: {acc} f1: {f1}\n")

    with open(os.path.join('pts', model_name+'_test_metrics.txt'), 'w') as file:
        file.write(json.dumps({'acc': acc, 'f1': f1}))

def predict_single_image(model, image_path:str, model_name:str="model.pt", image_size=224):
    '''
    Predicts the class belonging to the image given the pre-trained model.

    Parameters:

        model: Model 

        image_path:str Path of the image to give it a label.
    '''
    
    # Check if the necessary files for classification are present.

    model_path = os.path.join("pts", model_name)
    if not os.path.isfile(model_path):
        raise Exception(f"There is no pre-trained model with the '{model_path}' path to be used for classification.")
    
    idx_name_path = os.path.join("pts", "train_index.txt")
    if not os.path.isfile(idx_name_path):
        raise Exception(f"There is no index mapping file with the '{idx_name_path}' path to be used for classification.")
    
    # load the image
    data_transform = get_test_data_transform(image_size=image_size)
    
    img = Image.open(image_path)
    img = data_transform(img).unsqueeze(0)

    # predict the class
    model.eval()
    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

    # load the class names
    class_names = {}
    with open(idx_name_path, 'r') as file:
        for line in file.readlines():
            values = line.split(' ')
            class_names.update({values[1]:values[0]})

    y = preds.squeeze().item()
    print (f"# Prediction: The image is rated {class_names[y]}, with a probability of {'x'}.")
    