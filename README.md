# St. George Detector

The present repository aims to provide an DL model to detect St. George on images.

## Results

Different parameters and optimizers were tested, as well as two pre-trained networks, VGG16 and ResNet5. Between the two, ResNet5 obtained the best results.


<center>

| Model    | learning rate | optim   | ACC    | F1     |
|----------|---------------|---------|--------|--------|
| ResNet50 |     0.0001    |   Adam  | 0.9815 | 0.9777 |
| ResNet50 |     0.0001    | RMSPorp | 0.9782 | 0.9733 |
|   VGG16  |     0.001     | Adagrad | 0.9056 | 0.9224 |

</center>


It was decided to use pre-trained models, because the styles of the photos varied: from stained glass, realistic and abstract paintings, statues, people, etc. 

<p align="center">
  <img src="images/im1.jpg" width="230" title="st george pichture">
  <img src="images/im2.jpg" width="230" alt="st george pichture">
  <img src="images/im3.jpg" width="230" alt="st george pichture">
</p>

These networks in their first layers are already tuned for the detection of low-level features, while the upper ones are used for the extraction of more abstract features such as objects.

Thus, since most of the St. George pictures are composed of a man in armor, riding a horse, with sword or spear killing a lizard or dragon, it is expected that the pre-trained network will better capture those features and relate them appropriately.


## Reproductivity

To reproduce the obtained results, it is first necessary to create an environment and install the dependencies. The repository is cloned first.

```shell
$ git clone https://github.com/mjason98/detgeorges.git
$ cd detgeorges
```

Then create the environment and install the dependencies with pip.

```shell
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

For the training data, a folder with the files 'georges.csv' and 'non_georges.csv' is expected with the path <path_to_train_files>. It is necessary not to use a folder with the name 'train' inside the repo directory as it is reserved for creating the training data.

For the test data, just unzip the file and use the path to run the program. Only two folders, with the names 'george' and 'no_george', can be in the path. This can be checked with the 'ls' command.

```shell
$ unzip <path_to_test_zip>/george_test_task.zip
$ ls -la george_test_task
.
..
george
no_george 
```

To train the model, with the parameters that gave the best result, execute the following command.

```shell
$ python main.py -t <path_to_train_files> -p george_test_task
```

At the end of its execution, it will show the metrics obtained in the test set. Pre-trained networks as well as training data are automatically downloaded.

### Detect St. George in a single image

To detect St. George in a single image, use the following command with the path to the image.

```shell
$ python main.py --notrain -i <image path>
```

The above command depends on having previously trained the model, if not, it is preferable to just use the '-i' parameter to train the model first and then perform future classifications.


```shell
$ python main.py -t <path_to_train_files> -p george_test_task -i <image path>
```