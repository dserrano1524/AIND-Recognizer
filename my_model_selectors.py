import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """
    def calc_bic(self, n):
        #BIC = -2 * logL + p * logN
        hmm_model = self.base_model(n)
        # L is the likelihood of the fitted model
        # http://hmmlearn.readthedocs.io/en/latest/api.html#hmmlearn.base._BaseHMM.score
        # Compute the log probability under the model.
        logL = hmm_model.score(self.X,self.lengths)
        # N is the number of data points -> log(N)
        logN = np.log(len(self.X))
        # p is the number of parameters
        # p = n_components*n_components + 2*n_components*n_features - 1
        # from https://stats.stackexchange.com/questions/12341/number-of-parameters-in-markov-model
        # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/2
        p = n ** 2 + 2 * hmm_model.n_features * n - 1
        BIC = -2*logL+(p*logN)
        return hmm_model, BIC

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        '''The hmmlearn library may not be able to train or score all models.
        Implement try/except contructs as necessary to eliminate non-viable models from consideration.'''
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        try:
            best_model = None
            best_score = float("Inf")
            # ange self.min_n_components and self.max_n_components +1 (inclusive)
            for n in range(self.min_n_components, self.max_n_components+1):
                hmm_model, BIC = self.calc_bic(n)
                #Model selection: The lower the AIC/BIC value the better the model
                #(onlycompare AIC with AIC and BIC with BIC values!).
                if BIC < best_score:
                    best_score = BIC
                    best_model = hmm_model

            return best_model
        except:
            return self.base_model(self.n_constant)
            #import sys
            #e = sys.exc_info()[0]
            #return e


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''
    def cal_dic(self,n):
        ''' Calcultates DIC score based on the formula given in the paper.
        DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i)) '''
        model_scores = 0
        hmm_model = self.base_model(n)
        for word, w_lengths in self.all_word_Xlengths.items():
            # We avoid j=i in the Sum
            if word != self.this_word:
                logL = hmm_model.score(word,w_lengths)
                model_scores += logL
        # According to the given formula, alpha = 1
        alpha = 1
        M = len(self.all_word_Xlengths)
        DIC = hmm_model.score(self.X, self.lengths) - ((model_scores* alpha)/(M-1))
        return hmm_model, DIC

        def cal_dic_v2(self,n):
            ''' Calcultates DIC by using the mean of the scores as suggested by
             the mentor. DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i)) '''
            model_scores = []
            hmm_model = self.base_model(n)
            for word, w_lengths in self.all_word_Xlengths.items():
                # We avoid j=i in the Sum
                if word != self.this_word:
                    logL = hmm_model.score(word,w_lengths)
                    model_scores.append(logL)
            DIC = hmm_model.score(self.X, self.lengths) - (np.mean(model_scores))
            return hmm_model, DIC

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        try:
            best_model = None
            best_score = float("-Inf")
            # ange self.min_n_components and self.max_n_components +1 (inclusive)
            for n in range(self.min_n_components, self.max_n_components+1):
                hmm_model, DIC = self.calc_dic(n)
                if DIC > best_score:
                    best_score = DIC
                    best_model = hmm_model
            return best_model
        except:
            return self.base_model(self.n_constant)
            #import sys
            #e = sys.exc_info()[0]
            #return e


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def cal_cv(self, n):
        ''' Calcs cross validation score using scikit-learn K-Fold'''
        model_scores = []
        model = None
        split_method = KFold()
        for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
            X, lengths = combine_sequences(cv_test_idx, self.sequences)
            model = self.base_model(n)
            model_scores.append(model.score(X,lengths))
        return model, np.mean(model_scores)

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        try:
            best_model = None
            best_score = float("Inf")
            # ange self.min_n_components and self.max_n_components +1 (inclusive)
            for n in range(self.min_n_components, self.max_n_components+1):
                hmm_model, CV = self.cal_cv(n)
                if CV < best_score:
                    best_score = CV
                    best_model = hmm_model
            return best_model
        except:
            return self.base_model(self.n_constant)
            #import sys
            #e = sys.exc_info()[0]
            #return e
