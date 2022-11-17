import argparse



def check_params(arg=None, params:dict={}):
    parse = argparse.ArgumentParser(description='St George Detector')

    parse.add_argument('-p', dest='predict', help='Image to classify', 
                       required=False, default="")

    parse.add_argument('--notrain', help='Do not train the model', 
					   required=False, action='store_true', default=False)
    
    returns = parse.parse_args(arg)

    image = returns.predict
    notrain = bool(returns.notrain)

    params.update({'image':image, 'notrain':notrain})
    
    return 1