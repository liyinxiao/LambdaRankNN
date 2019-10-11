import unittest
import numpy as np
from LambdaRankNN import LambdaRankNN, RankNetNN

class GenericFieldsTestCase(unittest.TestCase):
    def test_pairwise_transform(self):
        X = np.array([[0.2, 0.3, 0.4],
                      [0.1, 0.7, 0.4],
                      [0.3, 0.4, 0.1],
                      [0.8, 0.4, 0.3],
                      [0.9, 0.35, 0.25]])
        y = np.array([0, 1, 0, 0, 2])
        qid = np.array([1, 1, 1, 2, 2])

        ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
        X1, X2, Y, weight = ranker._transform_pairwise(X, y, qid)

        self.assertTrue((X1 == np.array([[0.1, 0.7, 0.4],
                                        [0.1, 0.7, 0.4],
                                        [0.8, 0.4, 0.3]])).all())
        self.assertTrue((X2 == np.array([[0.2 , 0.3 , 0.4],
                                        [0.3 , 0.4 , 0.1],
                                        [0.9 , 0.35, 0.25]])).all())
        self.assertTrue((Y == np.array([1, 1, 0])).all())
        self.assertTrue((weight == np.array([0.36907024642854247, 0.13092975357145753, 0.36907024642854247])).all())
