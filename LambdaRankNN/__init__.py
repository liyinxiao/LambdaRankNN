
from keras import backend as K
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model
import numpy as np
import math


class RankerNN(object):

    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=('relu',), solver='adam'):
        """
        Parameters
        ----------
        input_size : integer
            Number of input features.
        hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
            The ith element represents the number of neurons in the ith
            hidden layer.
        activation : tuple, length = n_layers - 2, default ('relu',)
            The ith element represents activation function in the ith
            hidden layer.
        solver : {'adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', adamax},
        default 'adam'
            The solver for weight optimization.
            - 'adam' refers to a stochastic gradient-based optimizer proposed
              by Kingma, Diederik, and Jimmy Ba
        """
        if len(hidden_layer_sizes) != len(activation):
            raise ValueError('hidden_layer_sizes and activation should have the same size.')
        self.model = self._build_model(input_size, hidden_layer_sizes, activation)
        self.model.compile(optimizer=solver, loss="binary_crossentropy")

    @staticmethod
    def _build_model(input_shape, hidden_layer_sizes, activation):
        """
        Build Keras Ranker NN model (Ranknet / LambdaRank NN).
        """
        # Neural network structure
        hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            hidden_layers.append(Dense(hidden_layer_sizes[i], activation=activation[i], name=str(activation[i]) + '_layer' + str(i)))
        h0 = Dense(1, activation='linear', name='Identity_layer')
        input1 = Input(shape=(input_shape,), name='Input_layer1')
        input2 = Input(shape=(input_shape,), name='Input_layer2')
        x1 = input1
        x2 = input2
        for i in range(len(hidden_layer_sizes)):
            x1 = hidden_layers[i](x1)
            x2 = hidden_layers[i](x2)
        x1 = h0(x1)
        x2 = h0(x2)
        # Subtract layer
        subtracted = Subtract(name='Subtract_layer')([x1, x2])
        # sigmoid
        out = Activation('sigmoid', name='Activation_layer')(subtracted)
        # build model
        model = Model(inputs=[input1, input2], outputs=out)
        return model

    @staticmethod
    def _CalcDCG(labels):
        sumdcg = 0.0
        for i in range(len(labels)):
            rel = labels[i]
            if rel != 0:
                sumdcg += ((2 ** rel) - 1) / math.log2(i + 2)
        return sumdcg

    def _fetch_qid_data(self, y, qid, eval_at=None):
        """Fetch indices, relevances, idcg and dcg for each query id.
        Parameters
        ----------
        y : array, shape (n_samples,)
            Target labels.
        qid: array, shape (n_samples,)
            Query id that represents the grouping of samples.
        eval_at: integer
            The rank postion to evaluate dcg and idcg.
        Returns
        -------
        qid2indices : array, shape (n_unique_qid,)
            Start index for each qid.
        qid2rel : array, shape (n_unique_qid,)
            A list of target labels (relevances) for each qid.
        qid2idcg: array, shape (n_unique_qid,)
            Calculated idcg@eval_at for each qid.
        qid2dcg: array, shape (n_unique_qid,)
            Calculated dcg@eval_at for each qid.
        """
        qid_unique, qid2indices, qid_inverse_indices = np.unique(qid, return_index=True, return_inverse=True)
        # get item releveance for each query id
        qid2rel = [[] for _ in range(len(qid_unique))]
        for i, qid_unique_index in enumerate(qid_inverse_indices):
            qid2rel[qid_unique_index].append(y[i])
        # get dcg, idcg for each query id @eval_at
        if eval_at:
            qid2dcg = [self._CalcDCG(qid2rel[i][:eval_at]) for i in range(len(qid_unique))]
            qid2idcg = [self._CalcDCG(sorted(qid2rel[i], reverse=True)[:eval_at]) for i in range(len(qid_unique))]
        else:
            qid2dcg = [self._CalcDCG(qid2rel[i]) for i in range(len(qid_unique))]
            qid2idcg = [self._CalcDCG(sorted(qid2rel[i], reverse=True)) for i in range(len(qid_unique))]
        return qid2indices, qid2rel, qid2idcg, qid2dcg


    def _transform_pairwise(self, X, y, qid):
        return None, None, None, None


    def fit(self, X, y, qid, batch_size=None, epochs=1, verbose=1, validation_split=0.0):
        """Transform data and fit model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        qid: array, shape (n_samples,)
            Query id that represents the grouping of samples.
        """
        X1_trans, X2_trans, y_trans, weight = self._transform_pairwise(X, y, qid)
        self.model.fit([X1_trans, X2_trans], y_trans, sample_weight=weight, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, validation_split=validation_split)
        self.evaluate(X, y, qid)

    def predict(self, X):
        """Predict output.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        Returns
        -------
        y_pred: array, shape (n_samples,)
            Model prediction.
        """
        ranker_output = K.function([self.model.layers[0].input], [self.model.layers[-3].get_output_at(0)])
        return ranker_output([X])[0].ravel()

    def evaluate(self, X, y, qid, eval_at=None):
        """Predict and evaluate ndcg@eval_at.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        qid: array, shape (n_samples,)
            Query id that represents the grouping of samples.
        eval_at: integer
            The rank postion to evaluate NDCG.
        Returns
        -------
        ndcg@eval_at: float
        """
        y_pred = self.predict(X)
        tmp = np.array(np.hstack([y.reshape(-1, 1), y_pred.reshape(-1, 1), qid.reshape(-1, 1)]))
        tmp = tmp[np.lexsort((-tmp[:, 1], tmp[:, 2]))]
        y_sorted = tmp[:, 0]
        qid_sorted = tmp[:, 2]
        ndcg = self._EvalNDCG(y_sorted, qid_sorted, eval_at)
        if eval_at:
            print('ndcg@' + str(eval_at) + ': ' + str(ndcg))
        else:
            print('ndcg: ' + str(ndcg))

    def _EvalNDCG(self, y, qid, eval_at=None):
        """Evaluate ndcg@eval_at.
        Calculated ndcg@n is consistent with ndcg@n- in xgboost.
        """
        _, _, qid2idcg, qid2dcg = self._fetch_qid_data(y, qid, eval_at)
        sumndcg = 0
        count = 0.0
        for qid_unique_idx in range(len(qid2idcg)):
            count += 1
            if qid2idcg[qid_unique_idx] == 0:
                continue
            idcg = qid2idcg[qid_unique_idx]
            dcg = qid2dcg[qid_unique_idx]
            sumndcg += dcg / idcg
        return sumndcg / count


class RankNetNN(RankerNN):

    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=('relu',), solver='adam'):
        super(RankNetNN, self).__init__(input_size, hidden_layer_sizes, activation, solver)

    def _transform_pairwise(self, X, y, qid):
        """Transform data into ranknet pairs with balanced labels for
        binary classification.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        qid: array, shape (n_samples,)
            Query id that represents the grouping of samples.
        Returns
        -------
        X1_trans : array, shape (k, n_feaures)
            Features of pair 1
        X2_trans : array, shape (k, n_feaures)
            Features of pair 2
        weight: array, shape (k, n_faetures)
            Sample weight lambda.
        y_trans : array, shape (k,)
            Output class labels, where classes have values {0, 1}
        """
        qid2indices, qid2rel, qid2idcg, _ = self._fetch_qid_data(y, qid)
        X1 = []
        X2 = []
        weight = []
        Y = []
        for qid_unique_idx in range(len(qid2indices)):
            if qid2idcg[qid_unique_idx] == 0:
                continue
            IDCG = 1.0 / qid2idcg[qid_unique_idx]
            rel_list = qid2rel[qid_unique_idx]
            qid_start_idx = qid2indices[qid_unique_idx]
            for pos_idx in range(len(rel_list)):
                for neg_idx in range(len(rel_list)):
                    if rel_list[pos_idx] <= rel_list[neg_idx]:
                        continue
                    # balanced class
                    if 1 != (-1) ** (qid_unique_idx + pos_idx + neg_idx):
                        X1.append(X[qid_start_idx + pos_idx])
                        X2.append(X[qid_start_idx + neg_idx])
                        weight.append(1)
                        Y.append(1)
                    else:
                        X1.append(X[qid_start_idx + neg_idx])
                        X2.append(X[qid_start_idx + pos_idx])
                        weight.append(1)
                        Y.append(0)
        return np.asarray(X1), np.asarray(X2), np.asarray(Y), np.asarray(weight)


class LambdaRankNN(RankerNN):

    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=('relu',), solver='adam'):
        super(LambdaRankNN, self).__init__(input_size, hidden_layer_sizes, activation, solver)

    def _transform_pairwise(self, X, y, qid):
        """Transform data into lambdarank pairs with balanced labels
        for binary classification.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Features.
        y : array, shape (n_samples,)
            Target labels.
        qid: array, shape (n_samples,)
            Query id that represents the grouping of samples.
        Returns
        -------
        X1_trans : array, shape (k, n_feaures)
            Features of pair 1
        X2_trans : array, shape (k, n_feaures)
            Features of pair 2
        weight: array, shape (k, n_faetures)
            Sample weight lambda.
        y_trans : array, shape (k,)
            Output class labels, where classes have values {0, 1}
        """
        qid2indices, qid2rel, qid2idcg, _ = self._fetch_qid_data(y, qid)
        X1 = []
        X2 = []
        weight = []
        Y = []
        for qid_unique_idx in range(len(qid2indices)):
            if qid2idcg[qid_unique_idx] == 0:
                continue
            IDCG = 1.0 / qid2idcg[qid_unique_idx]
            rel_list = qid2rel[qid_unique_idx]
            qid_start_idx = qid2indices[qid_unique_idx]
            for pos_idx in range(len(rel_list)):
                for neg_idx in range(len(rel_list)):
                    if rel_list[pos_idx] <= rel_list[neg_idx]:
                        continue
                    # calculate lambda
                    pos_loginv = 1.0 / math.log2(pos_idx + 2)
                    neg_loginv = 1.0 / math.log2(neg_idx + 2)
                    pos_label = rel_list[pos_idx]
                    neg_label = rel_list[neg_idx]
                    original = ((1 << pos_label) - 1) * pos_loginv + ((1 << neg_label) - 1) * neg_loginv
                    changed = ((1 << neg_label) - 1) * pos_loginv + ((1 << pos_label) - 1) * neg_loginv
                    delta = (original - changed) * IDCG
                    if delta < 0:
                        delta = -delta
                    # balanced class
                    if 1 != (-1) ** (qid_unique_idx + pos_idx + neg_idx):
                        X1.append(X[qid_start_idx + pos_idx])
                        X2.append(X[qid_start_idx + neg_idx])
                        weight.append(delta)
                        Y.append(1)
                    else:
                        X1.append(X[qid_start_idx + neg_idx])
                        X2.append(X[qid_start_idx + pos_idx])
                        weight.append(delta)
                        Y.append(0)
        return np.asarray(X1), np.asarray(X2), np.asarray(Y), np.asarray(weight)
