import sklearn.datasets
import sklearn.ensemble
import sklearn.inspection
import sklearn.model_selection

# Set a seed for reproducibility

seed = 42

# Get iris data

dataset = sklearn.datasets.load_iris()

# Evaluate our classifier on the train and test data

def estimate_significance_of_permutation_test(classifier):
    """
    Compute permutation test score along with p-value.
    
    Source: <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html#sklearn-model-selection-permutation-test-score>
    """
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

estimate_significance_of_permutation_test(sklearn.dummy.DummyClassifier(random_state=seed))

classifier = sklearn.ensemble.HistGradientBoostingClassifier(random_state=seed)
estimate_significance_of_permutation_test(classifier)

# > What can you conclude about:
# > - the existence of a significant statistical association between the iris type and the input features (petal and sepal width and length)?
# > - the ability of each kind of estimator to assess or not such a statistical association between features and target variable?
#
# Since there are 3 output classes (virginica, setosa and versicolor), a random classifier would on average get the right prediction one third of the time. Accordingly, we see that the `DummyClassifier` is correct about 33% of the time, and that the permutation score is also about 33%. Since the `DummyClassifier` doesn't perform better than a random classifier, we can conclude that even if there were an association between the inputs and the output, it would not detect it.
#
# By contrast, based on the small $p$-value, it seems that the `HistGradientBoostingClassifier` model has detected a relationship between the input features and the output. Indeed, the classifier performs much better on the true data than on permuted data (95% accuracy against 33%). Hence it seems that this kind of classifier has a higher ability than `DummyClassifier` to uncover statistical relationships.
# Finally, according to the `scikit-learn` library, the interpretation of this $p$-value is:
# > The p-value [...] approximates the probability that the score would be obtained by chance.
#
# By this interpretation, it is unlikely that the input features and output are uncorrelated.
