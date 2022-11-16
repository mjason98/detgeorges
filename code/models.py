from ntpath import join
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os, time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=model.device))

def save_model(model, path):
    torch.save(model.state_dict(), path) 

def create_model_and_optimizer(model:str, lr=0.001, momentum=0.9):
    '''
    Creates models and optimizers

    Parameters:

        model:str the arquitecture of the model: ['VGG16', ...]

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
        exp_lr_scheduler = optim.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        return model_ft, optimizer_ft, exp_lr_scheduler
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
            epoch_acc = running_corrects.double() / dataset_size
            
            epoch_stats[phase].append((epoch_loss, epoch_acc))

            print (f"{phase} loss {epoch_loss} acc {epoch_acc}")
            
            #  best model
            if epoch_acc > best_acc and phase == 'val':
                best_acc = epoch_acc
                save_model(model, save_path)

        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return epoch_stats
