# Cat-recognition-app
A cloud-based machine learning application recognizing cats in pictures.

This repository contain all the necessary python code required to build a ML application
and deploy it on the AWS end-to-end.

## Demo

<img src="images/cover.png" alt="Cover" width="50%"/>


## Introduction
Although there are already lots of good tutorials telling you how to build a machine learning model,
I feel that there is little explanation about how to actually deploy your model as a web application.


So I decided to build a simple image classifier
that is able to recognize cats and deploy it using AWS Lambda in order to simulate(or at least practice)
how to actually deploy a ML model in real world.



## Steps to follow
- Build environment (on mac)
- Train a Convolutional Neural Network as image classifier
- Build a Flask application
    * Allow users upload images
    * Predict whether the images are cats using model trained previously
- Deploy the application on AWS




## Build environment (on mac)
Use python 3.6 to ensure that we can deploy our model on AWS Lambda later.
```commandline
pyenv install 3.6.1
```

Create a new virtual environment to manage dependencies
and use the env under current project folder.
```commandline
pyenv virtualenv 3.6.1 py3.6-ml-app
cd cat-recognition-app/
pyenv local py3.6-ml-app
```

Install libraries for training models and visualization.
We will train our models using TensorFlow on jupyter notebook.
```commandline
pip install numpy tensorflow jupyter scipy pillow matplotlib seaborn jupyter_contrib_nbextensions ipywidgets
```



## Train a Convoluational Neural Network

In this part, we will train a CNN to classify cats' images from dogs' image
using Kaggle dataset [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data). We will do the following things:
- Load, resize and normalize the images
- Create training/valid set
- Train a CNN model
- Serialize the model for later deployment

All steps described above will be included in the notebook [cat_recognizer](cat_recognizer.ipynb).
If you want to execute the code in the notebook, install all the extra dependencies.

```commandline
jupyter nbextension enable --py widgetsnbextension
```

Start a jupyter server.

```commandline

jupyter notebook
```

And you should be able to open and run the notebook at localhost:8888.

## Build a Flask application

In this part, we will build a simple flask web application which allow users
to upload images and predict whether there are cats in the images using the
model we trained in previous part.

We will need extra dependencies for the application:
```commandline
pip install flask flask-bootstrap boto3 zappa
```

To start the flask application:

```commandline
python app.py
```

And you should be able to view the app at localhost:5000 using the browsers.





## Deploy the application on AWS

We will deploy our model on AWS using AWS Lambda.
Again, extra dependencies for deploying the application.

To be continued.


## Miscellaneous
- [pyenv build fail](https://github.com/pyenv/pyenv/issues/655): Try install CLI dev tools
```commandline
xcode-select --install
```

- Dependencies

To install all the dependencies listed in [requirements.txt](requirements.txt)
all at once:

```commandline
pip install -r requirements.txt
```
