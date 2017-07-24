## Train
To make vocab file and training file, run `make_train_data.py`
```
$ python make_train_data.py
```

Start training the model using `train.py`, for example

```
$ python train.py
```

## Generate Sentence
```
$ python generate.py
```

The generated texts are changed when you use `seed` parameter.

```
$ python generate.py --seed=1234
```

