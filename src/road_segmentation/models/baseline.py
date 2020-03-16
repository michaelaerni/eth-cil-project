import sklearn
import sklearn.linear_model


def create_old_logreg_model(seed: int) -> sklearn.linear_model.LogisticRegression:
    return sklearn.linear_model.LogisticRegression(
        C=100000.0,
        class_weight='balanced',
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        max_iter=100,
        multi_class='ovr',
        n_jobs=1,
        penalty='l2',
        random_state=seed,
        tol=0.0001,
        verbose=0,
        warm_start=False
    )
