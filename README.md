# advml_mnist_subs

Final project for CSE 891: Adversarial Machine Learning

[Final paper](891_AA_with_FashionMNIST.pdf)

## Stages of accomplishment
1. Evaluate existing model (if available) on MNIST and compare to empirical paper results
2. Plug in Fashion-MNIST and train a new Pytorch model using given code
3. Evaluate new model using IDENTICAL procedure as step 1
4. OPTIONALLY repeat 2/3 using Kuzushiji-MNIST (cursive Japanese) - Not enough time


## Commands to train models
Don't need to add time to command when running on Azure ML Studio, creating an experiment shows run time and more useful stats like output logs

    $ python train_trades_mnist.py

    $ python train_trades_fashion_mnist.py

    $ python train.py --config config/mnist_crown.json

    $ python train.py --config config/fashion_mnist_crown_ibp.json


### Timing
|Attack|Training time|
|---|---|
|TRADES MNIST | 2h7m28s|
|TRADES Fashion MNIST | 2h6m25s|
|CROWN MNIST | 2h28m58s|
|CROWN Fashion MNIST | 2h19m28s|
|TRADES MNIST with AA | 2h18m9s|
|TRADES Fashion MNIST with AA | 2h17m41s|
|CROWN MNIST with AA | . | 
|CROWN Fashion MNIST with AA | . |
