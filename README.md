# Cat-recognition-app
A cloud-based machine learning (ML) application recognize cats in pictures.

This repository contain all the necessary python code required to build a ML application
and deploy it on the AWS.

## Demo




## Introduction
No matter how good you think your machine learning models are, if you can 
not expose it to the public, you won't be able to know how actually users are
going to interact with it. With this in mind, I decided to build a simple image classifier
that is able to recognize cats and deploy it using AWS Lambda in order to simulate how to
deploy a ML model in real world.
 


## Steps to follow
- Build environment (on mac)
- Train a Convoluational Neural Network(CNN)
- Deploy trained model



## Build environment (on mac)
Use python 3.6 to ensure that we can deploy our model on AWS Lambda later.
```commandline
pyenv install 3.6.1
```

Create a new virtual environment to manage/minimize dependencies and use the env under current project folder.
```commandline
pyenv virtualenv 3.6.1 py3.6-ml-app
cd cat-recognition-app/
pyenv local py3.6-ml-app
```

Install libraries for training models and visualization.
```commandline
pip install numpy tensorflow jupyter matplotlib seaborn tqdm
```

- Train a Convoluational Neural Network


## Miscellaneous
- [pyenv build fail](https://github.com/pyenv/pyenv/issues/655): Try install CLI dev tools
```commandline
xcode-select --install
```












