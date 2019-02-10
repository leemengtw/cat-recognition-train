# Cat-recognition-train
This repository demonstrates how to train a cat vs dog recognition model and export the model to an optimized frozen graph easy for deployment using TensorFlow.

## Requirements
- Python3 (Tested on 3.6.8)
- TensorFlow (Tested on 1.12.0)
- NumPy (Tested on 1.15.1)
- tqdm (Tested on 4.29.1)
- Dogs vs. Cats dataset from [https://www.kaggle.com/c/dogs-vs-cats](https://www.kaggle.com/c/dogs-vs-cats)
- (Optional if you want to run tests) PyTorch (Tested on 1.0.0 and 1.0.1)

## Build environment
We recommend using Anaconda3 / Miniconda3 to manage your python environment.

If the machine you're using does not have a GPU instance, you can just:
```
$ pip install -r requirements.txt
```
or,
```
$ conda install --file requirements.txt
```

However, if you want to use GPU to accelerate the training process, please visit [https://www.tensorflow.org/install/gpu](https://www.tensorflow.org/install/gpu) for more information.

## Train a Convoluational Neural Network

In this part, we will use [TensorFlow](https://github.com/tensorflow/tensorflow) to train a CNN to classify cats' images from dogs' image
using Kaggle dataset [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data). We will do the following things:
- Create training/valid set (dataset.py)
- Load, augment, resize and normalize the images using `tensorflow.data.Dataset` api. (dataset.py)
- Define a CNN model (net.py)
    * Here we use the [ShufflenetV2 structure](https://arxiv.org/abs/1807.11164), which achieves great balance between speed and accuracy.
    * We do transfer learning on ShuffleNetV2 using the pretrained weights from [https://github.com/ericsun99/Shufflenet-v2-Pytorch](https://github.com/ericsun99/Shufflenet-v2-Pytorch).
    * If you want to know how to load PyTorch weights onto TensorFlow model graph, please check `convert_pytorch_weight_test` starting from line 44 in `module_tests.py`.
- Train the CNN model (train.py)
- Serialize the model for deployment (train.py)

If you want to execute the code, make sure you have all package requirements installed, and Dogs vs. Cats training dataset placed in `datasets`. The folder structure should be like:

```
cat-recognition-train
+-- train.py
+-- net.py
+-- dataset.py
+-- datasets
    +-- train
    |   +-- cat.0.jpg
    |   +-- cat.1.jpg
    |   ...
    |   +-- cat.12499.jpg
    |   +-- dog.0.jpg
    |   +-- dog.1.jpg
    |   ...
    |   +-- dog.12499.jpg
+-- ...
```

After all requirements set, run:
```
$ python train.py
```

See `train.py` for available arguments.

## Visualizing Learning using Tensorboard
During training, you can supervise how is the training going by running:
```
$ tensorboard --logdir runs
```
And you can check the tensorboard summaries on `localhost:6006`.


When training the model defined in the [cat_recognizer](cat_recognizer.ipynb), in addition to the reported accuracy messages showed in the notebook, you may be wondering:
- how do our neural network look like?
- what kind of images do we actually put into the model?
- do model improve during training?

These questions can be answered or better understood by viewing [Tensorboard](https://www.tensorflow.org/get_started/summaries_and_tensorboard). To open the tensorboard for this repo, enter:

```commandline
tensorboard --logdir=/tmp/tensorboard/cat-recognizer/
```

And you should be able to see all the interesting things on `localhost:6006`:

### Neural Network structure

As shown below, our simple neural network consist of two conv layers, followed by
one fully-connected layer (fc1) and the output layer (fc2) with single neuron.
In order to prevent overfitting, there is also a dropout mechnism between conv layer
and fully-connected layer.

<p align="center">
  <img src="images/first_naive_nn.png">
</p>

Notice here for the sake of clarity, some nodes (e.g. save, evaluation) are
removed so that only the training nodes remains. You may see a more complex
compuation graph on Tensorboard.

### Model Performance

<p align="center">
  <img src="images/scalars_on_tensor_board.png" >
  <caption>Accuracy and loss of trained model on Tensorboard</caption>
</p>


### Some images used for Training

<p align="center">
  <img src="images/training_images_on_tensorboard.png" >
  <caption>Images used in a mini-batch</caption>
</p>



## Build a Flask application

In this part, we will build a simple flask web application which allow users
to upload images and predict whether there are cats in the images using the
model we trained in previous part.

In order to run the app, extra dependencies are needed:
```commandline
pip install flask flask-bootstrap boto3
```

To start the flask application:

```commandline
python app.py
```

And you should be able to view the app at localhost:5000 using the browsers.


## Deploy the application on Heroku

In order to deploy the app on the Heroku, a user account and the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) are required.

To install the Heroku CLI on mac:
```commandline
brew install heroku/brew/heroku
```

Login using your account:
```commandline
heroku login
```

Install the dependencies and setting files:
```commandline
pip install gunicorn
pip freeze > requirements.txt

touch runtime.txt
echo "python-3.6.1" > runtime.txt

touch Procfile
echo "web: gunicorn app:app --log-file=-" > Procfile
```

Create a new Heroku application:
```commandline
heroku create

Creating app... done, ⬢ damp-anchorage-60936
https://damp-anchorage-60936.herokuapp.com/ | https://git.heroku.com/damp-anchorage-60936.git
```

Deploy the application on Heroku. Your application id will be different from ours, which is `damp-anchorage-60936`
```
heroku git:remote -a damp-anchorage-60936
git add .
git commit -m "First commit"
git push heroku master
```

And you should be able to see the application on `https://YOUR-APPLICATION-NUM.herokuapp.com/`.


## Run app on Docker
Get the image on the Dockerhub
```commandline
docker pull leemeng/cat
```

Enable access to application running on docker @ `localhost:1234`
```commandline
docker run -dp 1234:5000 leemeng/cat
```

Check whether the application is running
```commandline
docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS                    NAMES
a887934e1849        leemeng/cat         "python3 app.py"    3 minutes ago       Up 3 minutes        0.0.0.0:1234->5000/tcp   thirsty_carson
```

Stop the application
```commandline
docker stop a887934e1849
```

## Trouble Shooting
### [pyenv build fail](https://github.com/pyenv/pyenv/issues/655)

Try install CLI dev tools
```commandline
xcode-select --install
```
### Incompatable ruby version when installing Heroku CLI (MAC)

Update ruby using brew and make it the default ruby
```commandline
brew install rbenv ruby-build

# Add rbenv to bash so that it loads every time you open a terminal
echo 'if which rbenv > /dev/null; then eval "$(rbenv init -)"; fi' >> ~/.bash_profile
source ~/.bash_profile

# Install Ruby
rbenv install 2.4.2
rbenv global 2.4.2
ruby -v
```

## Dependencies

To install all the dependencies listed in [requirements.txt](requirements.txt)
all at once:

```commandline
pip install -r requirements.txt
```
