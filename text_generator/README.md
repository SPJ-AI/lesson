## Train
To make vocab file and training file, run `make_train_data.py`
```
$ python3 make_train_data.py
```

Start training the model using `train.py`, for example

```
$ python3 train.py
```

## Generate Sentence
```
$ python3 generate.py
```

You can change the generated texts randomly by using `seed` parameter.

```
$ python3 generate.py --seed=1234
```

