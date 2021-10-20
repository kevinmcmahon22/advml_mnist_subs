# advml_mnist_subs


## Stages of accomplishment
1. Evaluate existing model (if available) on MNIST and compare to empirical paper results
2. Plug in Fashion-MNIST and train a new Pytorch model using given code
3. Evaluate new model using IDENTICAL procedure as step 1
4. OPTIONALLY repeat 2/3 using Kuzushiji-MNIST (cursive Japanese)


## COMMANDS TO TRAIN MODELS
add time in front when running on Azure to see timing
    NO NEED, creating an experiment shows run time and many more useful stats like output logs

$ python train_trades_mnist.py
$ python train_trades_fashion_mnist.py

$ python train.py --config config/mnist_crown.json
$ python train.py --config config/fashion_mnist_crown_ibp.json


## Notes
Running produces .log files in mnist_crown directory

create new config file for fashion mnist, copy mnist_crown.json and change dataset to fashion-mnist

python train.py --config config/fashion_mnist_crown_ibp.json

CROWN-IBP datasets stored in advml_mnist_subs/CROWN-IBP/data directory
    had to download data twice (once for each of my attacks) but this was better compared to messing with code further

After downloading pt files, either run in Azure, send to Tyler for GPU usage, or run on my PC using CPU
    If I send to Tyler make sure to uncomment CUDA lines
    Azure seems like the best bet

Crete Azure free account, azure machine learning is always free
create azureML workspace and upload this git repo to the notebooks interface
    There I can assign compute instances and run python scripts to train models
    First I tried 4 cores, 14gb ram, 28gb disk, 0.23/hr
    provides access to a linux terminal, time works

Current Azure compute is way too slow, took almost 5 minutes for 1/5 of an epoch, need 100 epochs
    look into using a more powerful compute/gpu support, remember I have 200 bucks of credits to spare
    probably too risky to let it sit and run for hours on end, pay a little money for convinience

Followed azure pytorch walkthrough, it actually made sense once I knew what was going on in Azure
    successfully compiled train_trades_mnist.py, low priority compute cluster (gpu) with 6 cores (quota is 10)

Connected GPU successfully, currently running epoch 1 of train_trades_mnist experiment
    this is taking some time but azure with GPU definately makes this process much faster than with my CPU
    maybe I can run experiments overnight to train models

currently using Tesla K80, switch to Tesla V100 for subsequent experiments
    0.61 vs 0.18, time is more valuable than money at this point in the project

If I have extra time I would like to do azure ML stuff from VSCode, for this project doing everything through the protal will have to do

Don't touch this git repo any more besides readme, keep all changes/most up to date code in Azure

might not need 100 epochs if TRADES uses an early stop condition

Downgrade compute (CPU) to B1S (burstable VMs) 

reverted CUDA comments in CROWN and created pytorch train crown script

change nuymber of epochs back to 100

I will create env myself via condaDependencies becuase I may need joblib to register/download model

I can't directly download model by running in vscode (which makes sense) so there's no reason to use vscode

the missing piece was joblib from sklearn help page https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-scikit-learn 

crown ibp takes 15 minutes or so to train 1 small model, must train 10 of them

Train 10 small MNIST classifiers with CROWN-IBP, 1 with TRADES
    let run in Azure ML Studio possibly overnight
    will have 22 models that I can test with autoattack (figure that out tomorrow)

I will have no payments since I have azure free account

the training logs were different for mnist and fash mnist trades, so that means different datasets were used

code used to train models can be viewed in snapshot of run of an experiment in ml studio

503 service not found error when downloading MNIST dataset via torchvision.datasets in autoattack script

Added autoattack to advml_subs folder on azure, am now able to access autoattack within docker image created for run of the attack experiment in ml studio

#### LESSONS
Sleep on it
Take exercise breaks
read documentation fully before getting frustrated

### Timing
TRADES MNIST - 2h7m28s
TRADES Fashion MNIST - 2h6m25s
CROWN MNIST - 2h28m58s
CROWN Fashion MNIST - 2h19m28s

TRADES MNIST with AA - 2h18m9s
TRADES Fashion MNIST with AA - 2h17m41s
CROWN MNIST with AA - 
CROWN Fashion MNIST with AA -
