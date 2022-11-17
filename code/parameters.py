import argparse
import os


def check_params(arg=None, params:dict={}):
    parse = argparse.ArgumentParser(description='St George Detector')

    parse.add_argument('-i', dest='predict', help='Image to classify', 
                       required=False, default="")
    
    parse.add_argument('-t', dest='train', help='Path to the folder with training data. This folder must contain two CSV files: \'georges.csv\' and \'non_georges.csv\'.', 
                       required=False, default="")
    
    parse.add_argument('-p', dest='test', help='Path to the folder with test data. This folder must contain two folders: \'george\' and \'no_george\'.', 
                       required=False, default="test/")
    
    parse.add_argument('-e', dest='epochs', help='Number of epochs.', 
                       required=False, default="20")

    parse.add_argument('-b', dest='batch', help='Batch size.',
                       required=False, default="32")

    parse.add_argument('--lr', dest='learningrate', help='Learning rate.',
                       required=False, default="0.0001")

    parse.add_argument('--mtype', dest='mtype', help='Model type.', 
					   required=False, default='RESNET', choices=['VGG16', 'RESNET'])                 

    parse.add_argument('--notrain', help='Do not train the model', 
					   required=False, action='store_true', default=False)
    
    returns = parse.parse_args(arg)

    image = returns.predict
    notrain = bool(returns.notrain)

    positive_examples = os.path.join(returns.train, "georges.csv")
    negative_examples = os.path.join(returns.train, "non_georges.csv")
    test_folder = returns.test
    epochs = int(returns.epochs)
    batch = int(returns.batch)
    lr = float(returns.learningrate)
    model_type = returns.mtype

    params.update({
        'image':image,
        'notrain':notrain,
        "positive_examples":positive_examples,
        "negative_examples":negative_examples,
        "test_folder":test_folder,
        "epochs":epochs,
        "batch_size":batch,
        "lr":lr,
        "model_type":model_type,
    })
    
    return 1