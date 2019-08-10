LambdaRankNN
==========

Python library for training pairwise Learning-To-Rank Neural Network models (RankNet NN, LambdaRank NN).

## Supported model structure

It supports pairwise Learning-To-Rank (LTR) algorithms such as Ranknet and LambdaRank, where the underlying model (hidden layers) is a neural network (NN) model. 

<img src="https://github.com/liyinxiao/LambdaRankNN/blob/master/assets/rankerNN2pmml_model.png" width=750 />

## Installation
```
pip install LambdaRankNN
```

## Example

Example on a LambdaRank NN model, with the training data below. 

<img src="https://github.com/liyinxiao/LambdaRankNN/blob/master/assets/query_data.png" width=350>

```python
import numpy as np
from LambdaRankNN import LambdaRankNN

# generate query data
X = np.array([[0.2, 0.3, 0.4],
              [0.1, 0.7, 0.4],
              [0.3, 0.4, 0.1],
              [0.8, 0.4, 0.3],
              [0.9, 0.35, 0.25]])
y = np.array([0, 1, 0, 0, 2])
qid = np.array([1, 1, 1, 2, 2])

# train model
ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
ranker.fit(X, y, qid, epochs=5)
y_pred = ranker.predict(X)
ranker.evaluate(X, y, qid, eval_at=2)
```

## Converting model to pmml

The trained model can be conveniently converted to pmml, with Python library rankerNN2pmml. 

```python
from rankerNN2pmml import rankerNN2pmml
params = {
    'feature_names': ['Feature1', 'Feature2', 'Feature3'],
    'target_name': 'score'
}

rankerNN2pmml(estimator=ranker.model, file='Model_example.xml', **params)
```
