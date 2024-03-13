import sklearn.datasets
import sklearn.ensemble
import sklearn.inspection
import sklearn.model_selection
import matplotlib.pyplot as plt
import typing

# Set a seed for reproducibility

seed = 42

# Get iris data and split it into a training set and a test set: we will train our classifier on the train set only, and evaluate it on the test set to see if the model is able to generalize to unseen data

dataset = sklearn.datasets.load_iris()

# Initialize our classifier and train it

classifier = sklearn.dummy.DummyClassifier(random_state=seed)

# Evaluate our classifier on the train and test data

def estimate_significance_of_permutation_test(classifier):
    score, permutation_scores, pvalue = sklearn.model_selection.permutation_test_score(
        classifier, dataset.data, dataset.target, random_state=seed
    )

    print(f"Test results for {classifier}")
    print(f"Original Score: {score:.3f}")
    print(
        f"Permutation Scores: {permutation_scores.mean():.3f} +/- "
        f"{permutation_scores.std():.3f}"
    )
    print(f"P-value: {pvalue:.3f}")#

estimate_significance_of_permutation_test(classifier)
    
classifier = sklearn.ensemble.HistGradientBoostingClassifier(random_state=seed)
estimate_significance_of_permutation_test(classifier)
