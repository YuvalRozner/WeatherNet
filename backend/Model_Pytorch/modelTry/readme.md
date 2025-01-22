# try model

first try of the model - not running yet

the idea: 

shared weights on 1 d cnn to extract local features of the region, + another path of 1d cnn to the target station. 

feed the input + incorpate the coordinates into multihead transformer in order to learn the spatial dependencies and long term dependencies.