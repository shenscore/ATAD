"""
The file containing implementations to all of the query strategies. References to all of these methods can be found in
the blog that accompanies this code.
"""

import gc
from scipy.spatial import distance_matrix

from keras.models import Model
import keras.backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Lambda
from keras import optimizers

from label.model import *

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np

#def initSampling(features,n=20):
#    kmeans = KMeans(n_clusters=n).fit(features)
#
#    labeled_idx = np.argmin(cdist(features,kmeans.cluster_centers_,'euclidean'),axis=0)
#    return labeled_idx

def get_unlabeled_idx(X_train, labeled_idx):
    """
    Given the training set and the indices of the labeled examples, return the indices of the unlabeled examples.
    """
    return np.arange(X_train.shape[0])[np.logical_not(np.in1d(np.arange(X_train.shape[0]), labeled_idx))]


class QueryMethod:
    """
    A general class for query strategies, with a general method for querying examples to be labeled.
    """

    def __init__(self, model, input_shape=(28,28), num_labels=10, gpu=1):
        self.model = model
        self.input_shape = input_shape
        self.num_labels = num_labels
        self.gpu = gpu

    def query(self, X_train, Y_train, labeled_idx, amount):
        """
        get the indices of labeled examples after the given amount have been queried by the query strategy.
        :param X_train: the training set
        :param Y_train: the training labels
        :param labeled_idx: the indices of the labeled examples
        :param amount: the amount of examples to query
        :return: the new labeled indices (including the ones queried)
        """
        return NotImplemented

    def update_model(self, new_model):
        del self.model
        gc.collect()
        self.model = new_model


class RandomSampling(QueryMethod):
    """
    A random sampling query strategy baseline.
    """

    # def __init__(self, model, input_shape, num_labels, gpu):
    #     super().__init__(model, input_shape, num_labels, gpu)
    def __init__(self, model, input_shape, num_labels,):
        super().__init__(model, input_shape, num_labels,)

    def query(self, X_train, Y_train, labeled_idx, amount, RandomSeed=False):
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        if RandomSeed:
            np.random.seed(RandomSeed)
        new_labeled_idx = np.random.choice(unlabeled_idx, amount, replace=False)
        # return np.hstack((labeled_idx, np.random.choice(unlabeled_idx, amount, replace=False)))
        return (labeled_idx,new_labeled_idx)


class UncertaintySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the minimal top confidence.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        predictions = self.model.predict(X_train[unlabeled_idx, :])

        unlabeled_predictions = np.amax(predictions, axis=1)

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class UncertaintyEntropySampling(QueryMethod):
    """
    The basic uncertainty sampling query strategy, querying the examples with the top entropy.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        predictions = self.model.predict(X_train[unlabeled_idx, :])

        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)

        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianUncertaintySampling(QueryMethod):
    """
    An implementation of the Bayesian active learning method, using minimal top confidence as the decision rule.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 20

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t,:,:] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        uncertainties = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        split = 128  # split into iterations of 128 due to memory constraints
        while i < unlabeled_idx.shape[0]:

            if i+split > unlabeled_idx.shape[0]:
                preds, unc = self.dropout_predict(X_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
                uncertainties[i:] = unc
            else:
                preds, unc = self.dropout_predict(X_train[unlabeled_idx[i:i+split], :])
                predictions[i:i+split] = preds
                uncertainties[i:i+split] = unc
            i += split

        unlabeled_predictions = np.amax(predictions, axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))


class BayesianUncertaintyEntropySampling(QueryMethod):
    """
    An implementation of the Bayesian active learning method, using maximal entropy as the decision rule.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.T = 100

    def dropout_predict(self, data):

        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        predictions = np.zeros((self.T, data.shape[0], self.num_labels))
        for t in range(self.T):
            predictions[t,:,:] = f([data, 1])[0]

        final_prediction = np.mean(predictions, axis=0)
        prediction_uncertainty = np.std(predictions, axis=0)

        return final_prediction, prediction_uncertainty

    def query(self, X_train, Y_train, labeled_idx, amount):

        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)

        predictions = np.zeros((unlabeled_idx.shape[0], self.num_labels))
        i = 0
        while i < unlabeled_idx.shape[0]: # split into iterations of 1000 due to memory constraints

            if i+1000 > unlabeled_idx.shape[0]:
                preds, _ = self.dropout_predict(X_train[unlabeled_idx[i:], :])
                predictions[i:] = preds
            else:
                preds, _ = self.dropout_predict(X_train[unlabeled_idx[i:i+1000], :])
                predictions[i:i+1000] = preds

            i += 1000

        unlabeled_predictions = np.sum(predictions * np.log(predictions + 1e-10), axis=1)
        selected_indices = np.argpartition(unlabeled_predictions, amount)[:amount]
        return np.hstack((labeled_idx, unlabeled_idx[selected_indices]))



# class DiscriminativeSampling(QueryMethod):
#     """
#     An implementation of DAL (discriminative active learning), using the raw pixels as the representation.
#     """

#     def __init__(self, model, input_shape, num_labels, gpu):
#         super().__init__(model, input_shape, num_labels, gpu)

#         self.sub_batches = 10

#     def query(self, X_train, Y_train, labeled_idx, amount):

#         # subsample from the unlabeled set:
#         unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
#         unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

#         # iteratively sub-sample using the discriminative sampling routine:
#         labeled_so_far = 0
#         sub_sample_size = int(amount / self.sub_batches)
#         while labeled_so_far < amount:
#             if labeled_so_far + sub_sample_size > amount:
#                 sub_sample_size = amount - labeled_so_far

#             model = train_discriminative_model(X_train[labeled_idx], X_train[unlabeled_idx], self.input_shape, gpu=self.gpu)
#             predictions = model.predict(X_train[unlabeled_idx])
#             selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
#             labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
#             labeled_so_far += sub_sample_size
#             unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
#             unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

#             # delete the model to free GPU memory:
#             del model
#             gc.collect()

#         return labeled_idx


class DiscriminativeRepresentationSamplingInteractive(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using the learned representation as our representation.
    This implementation is the one which performs best in practice.
    """

    def __init__(self, model, input_shape, num_labels,):
        super().__init__(model, input_shape, num_labels,)

        self.sub_batches = 1
    # def __init__(self, model, input_shape, num_labels, gpu):
    #     super().__init__(model, input_shape, num_labels, gpu)

    #     self.sub_batches = 20

    def query(self, X_train, Y_train, labeled_idx, amount):

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        # unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)
        unlabeled_idx = np.random.choice(unlabeled_idx, np.min([10000, unlabeled_idx.size]), replace=False)

        # # get current predictions
        # pred    = self.model.predict(X_train[unlabeled_idx])
        # posPred_idx = unlabeled_idx[(np.argmax(pred,axis=1)==1).nonzero()[0]]
        # negPred_idx = unlabeled_idx[(np.argmax(pred,axis=1)==0).nonzero()[0]]

        # pos_unlabeled_idx = np.random.choice(posPred_idx, np.min([5000, posPred_idx.size]), replace=False)
        # neg_unlabeled_idx = np.random.choice(negPred_idx, np.min([5000, negPred_idx.size]), replace=False)
        # # pos_unlabeled_idx = np.random.choice(posPred_idx, np.min([labeled_idx.shape[0]*5, posPred_idx.size]), replace=False)
        # # neg_unlabeled_idx = np.random.choice(negPred_idx, np.min([labeled_idx.shape[0]*5, negPred_idx.size]), replace=False)
        # unlabeled_idx = np.hstack((pos_unlabeled_idx,neg_unlabeled_idx))


        embedding_model = Model(inputs=self.model.input,
                                outputs=self.model.get_layer('softmax').input)
        representation = embedding_model.predict(X_train, batch_size=128).reshape((X_train.shape[0], -1, 1))

        # iteratively sub-sample using the discriminative sampling routine:
        sub_sample_size = amount
        new_labeled_idx = np.array([])

        model = train_discriminative_model(representation[labeled_idx], representation[unlabeled_idx], representation[0].shape, gpu=self.gpu)
        predictions = model.predict(representation[unlabeled_idx])
        selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
        labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
        new_labeled_idx = np.hstack((new_labeled_idx, unlabeled_idx[selected_indices]))

        # delete the model to free GPU memory:
        del model
        gc.collect()
        del embedding_model
        gc.collect()

        return (labeled_idx,new_labeled_idx)

    # def query(self, X_train, Y_train, labeled_idx, amount):

    #     # subsample from the unlabeled set:
    #     unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
    #     unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

    #     embedding_model = Model(inputs=self.model.input,
    #                             outputs=self.model.get_layer('softmax').input)
    #     representation = embedding_model.predict(X_train, batch_size=128).reshape((X_train.shape[0], -1, 1))

    #     # iteratively sub-sample using the discriminative sampling routine:
    #     labeled_so_far = 0
    #     sub_sample_size = int(amount / self.sub_batches)
    #     new_labeled_idx = np.array([])
    #     while labeled_so_far < amount:
    #         if labeled_so_far + sub_sample_size > amount:
    #             sub_sample_size = amount - labeled_so_far

    #         model = train_discriminative_model(representation[labeled_idx], representation[unlabeled_idx], representation[0].shape, gpu=self.gpu)
    #         predictions = model.predict(representation[unlabeled_idx])
    #         selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
    #         labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
    #         new_labeled_idx = np.hstack((new_labeled_idx, unlabeled_idx[selected_indices]))
    #         labeled_so_far += sub_sample_size
    #         unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
    #         unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

    #         # delete the model to free GPU memory:
    #         del model
    #         gc.collect()
    #     del embedding_model
    #     gc.collect()

    #     return (labeled_idx,new_labeled_idx)
class DiscriminativeRepresentationSampling(QueryMethod):
    """
    An implementation of DAL (discriminative active learning), using the learned representation as our representation.
    This implementation is the one which performs best in practice.
    """

    def __init__(self, model, input_shape, num_labels, gpu):
        super().__init__(model, input_shape, num_labels, gpu)

        self.sub_batches = 20


    def query(self, X_train, Y_train, labeled_idx, amount):

        # subsample from the unlabeled set:
        unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
        unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

        embedding_model = Model(inputs=self.model.input,
                                outputs=self.model.get_layer('softmax').input)
        representation = embedding_model.predict(X_train, batch_size=128).reshape((X_train.shape[0], -1, 1))

        # iteratively sub-sample using the discriminative sampling routine:
        labeled_so_far = 0
        sub_sample_size = int(amount / self.sub_batches)
        while labeled_so_far < amount:
            if labeled_so_far + sub_sample_size > amount:
                sub_sample_size = amount - labeled_so_far

            model = train_discriminative_model(representation[labeled_idx], representation[unlabeled_idx], representation[0].shape, gpu=self.gpu)
            predictions = model.predict(representation[unlabeled_idx])
            selected_indices = np.argpartition(predictions[:,1], -sub_sample_size)[-sub_sample_size:]
            labeled_idx = np.hstack((labeled_idx, unlabeled_idx[selected_indices]))
            labeled_so_far += sub_sample_size
            unlabeled_idx = get_unlabeled_idx(X_train, labeled_idx)
            unlabeled_idx = np.random.choice(unlabeled_idx, np.min([labeled_idx.shape[0]*10, unlabeled_idx.size]), replace=False)

            # delete the model to free GPU memory:
            del model
            gc.collect()
        del embedding_model
        gc.collect()

        return labeled_idx

