import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    for word, (X, lengths) in test_set.get_all_Xlengths().items():
        word_probabilities = dict()
        best_guess_word = ""
        best_model_score = float('-Inf')
        # According to SinglesData, load_data already inserts words ordered by word_id
        for word, model in models.items():
            try:
                log_likelihood = model.score(X,lengths)
                word_probabilities[word] = log_likelihood
                if log_likelihood > best_model_score:
                    best_model_score = log_likelihood
                    best_guess_word = word
            except:
                word_probabilities['Error_in word:_' + word] = float('-Inf')
        probabilities.append(word_probabilities)
        guesses.append(best_guess_word)

    return probabilities, guesses
