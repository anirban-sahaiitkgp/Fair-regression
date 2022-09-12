from sklearn.model_selection import KFold
import cvxpy as cp
import numpy as np
from src.fairness_penalty import fairness_penalty, fairness_penalty_lin


def logloss(X, y, w):
    """
    Computes the logloss for logistic regression in single model setting
    :param X: featuers
    :param y: ground-truth labels
    :param w: regression weight vector
    :return: logloss for logistic regression in single model setting
    """

    return -(1 / X.shape[0]) * cp.sum(cp.multiply(y, -cp.logistic(-X @ w)) + cp.multiply((1 - y), -cp.logistic(X @ w)))


def logloss_sep(indices1, indices2, X, y, w1, w2):
    """
    Computes the logloss for logistic regression in separate model setting
    :param indices1: the indices of X that fall in protected group 1
    :param indices2: the indices of X that fall in protected group 2
    :param X: featuers
    :param y: ground-truth labels
    :param w1: regression weight vector for group 1
    :param w2: regression weight vector for group 2
    :return: logloss for logistic regression in separate model setting
    """
    i1 = np.where(indices1)[0]
    i2 = np.where(indices2)[0]
    S1 = X[i1]
    y1 = y[i1]
    S2 = X[i2]
    y2 = y[i2]

    return -(1 / X.shape[0]) * (cp.sum(cp.multiply(y1, -cp.logistic(-S1 @ w1)) + cp.multiply((1 - y1), -cp.logistic(S1 @ w1))) + cp.sum(cp.multiply(y2, -cp.logistic(-S2 @ w2)) + cp.multiply((1 - y2), -cp.logistic(S2 @ w2))))


def meanse(X, y, w):
    """
    Computes the MSE for linear regression in single model setting
    :param X: featuers
    :param y: ground-truth labels
    :param w: regression weight vector
    :return: MSE for linear regression in single model setting
    """

    return (1 / X.shape[0]) * cp.sum_squares(X @ w - y)


def meanse_sep(indices1, indices2, X, y, w1, w2):
    """
    Computes the MSE for linear regression in separate model setting
    :param indices1: the indices of X that fall in protected group 1
    :param indices2: the indices of X that fall in protected group 2
    :param X: featuers
    :param y: ground-truth labels
    :param w1: regression weight vector for group 1
    :param w2: regression weight vector for group 2
    :return: MSE for linear regression in separate model setting
    """

    i1 = np.where(indices1)[0]
    i2 = np.where(indices2)[0]
    S1 = X[i1]
    y1 = y[i1]
    S2 = X[i2]
    y2 = y[i2]
    return (1 / X.shape[0]) * (cp.sum_squares(S1 @ w1 - y1) + cp.sum_squares(S2 @ w2 - y2))


def l2norm(w):
    """
    Computes l2-norm of weight vector
    :param w: regression weight vector
    :return: l2-norm of w
    """

    return (cp.norm(w, 2))**2


def get_loss(X, y, idx1, idx2, g, ll):
    """
    Computes the total loss values for the given value of lambda in logistic regression setting
    :param X: pandas dataframe containing the features
    :param y: pandas dataframe containing the ground-truth labels
    :param idx1: the indices of X that fall in protected group 1
    :param idx2: the indices of X that fall in protected group 2
    :param g: value of gamma
    :param ll: value of lambda
    :return:
    """

    L = [0, 0, 0, 0, 0, 0]
    lambd = cp.Parameter(nonneg=True)
    gamma = cp.Parameter(nonneg=True)
    # 10-fold cross validation for picking best gamma for given lambda
    cv = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        w1 = cp.Variable((X_train.shape[1], 1))
        w2 = cp.Variable((X_train.shape[1], 1))
        w3 = cp.Variable((X_train.shape[1], 1))
        w1sep1 = cp.Variable((X_train.shape[1], 1))
        w1sep2 = cp.Variable((X_train.shape[1], 1))
        w2sep1 = cp.Variable((X_train.shape[1], 1))
        w2sep2 = cp.Variable((X_train.shape[1], 1))
        w3sep1 = cp.Variable((X_train.shape[1], 1))
        w3sep2 = cp.Variable((X_train.shape[1], 1))
        penal = fairness_penalty(idx1[train_index], idx2[train_index], w1, w2, w3, w1sep1, w1sep2, w2sep1, w2sep2, w3sep1, w3sep2, X_train, y_train)

        # minimize the objective function for individual fairness - single model
        problem_individual = cp.Problem(cp.Minimize(logloss(X_train, y_train, w1) + lambd * penal['individual'] + gamma * l2norm(w1)))        
        lambd.value = ll
        gamma.value = g
        problem_individual.solve()

        # minimize the objective function for group fairness - single model
        problem_group = cp.Problem(cp.Minimize(logloss(X_train, y_train, w2) + lambd * penal['group'] + gamma * l2norm(w2)))
        lambd.value = ll
        gamma.value = g
        problem_group.solve()

        # minimize the objective function for hybrid fairness - single model
        problem_hybrid = cp.Problem(cp.Minimize(logloss(X_train, y_train, w3) + lambd * penal['hybrid'] + gamma * l2norm(w3)))
        lambd.value = ll
        gamma.value = g
        problem_hybrid.solve()

        # minimize the objective function for individual fairness - separate model
        problem_individualsep = cp.Problem(cp.Minimize(logloss_sep(idx1[train_index], idx2[train_index], X_train, y_train, w1sep1, w1sep2) + lambd * penal['individualsep'] + gamma * (l2norm(w1sep1) + l2norm(w1sep2))))
        problem_individualsep.solve()

        # minimize the objective function for group fairness - separate model
        problem_groupsep = cp.Problem(cp.Minimize(logloss_sep(idx1[train_index], idx2[train_index], X_train, y_train, w2sep1, w2sep2) + lambd * penal['groupsep'] + gamma * (l2norm(w2sep1) + l2norm(w2sep2))))
        problem_groupsep.solve()

        # minimize the objective function for hybrid fairness - separate model
        problem_hybridsep = cp.Problem(cp.Minimize(logloss_sep(idx1[train_index], idx2[train_index], X_train, y_train, w3sep1, w3sep2) + lambd * penal['hybridsep'] + gamma * (l2norm(w3sep1) + l2norm(w3sep2))))
        problem_hybridsep.solve()

        # get loss values in test set
        floss = fairness_penalty(idx1[test_index], idx2[test_index], w1.value, w2.value, w3.value, w1sep1.value, w1sep2.value, w2sep1.value, w2sep2.value, w3sep1.value, w3sep2.value, X_test, y_test)
        L[0] += (logloss(X_test, y_test, w1.value)).value + lambd.value * (floss['individual']).value
        L[1] += (logloss(X_test, y_test, w2.value)).value + lambd.value * (floss['group']).value
        L[2] += (logloss(X_test, y_test, w3.value)).value + lambd.value * (floss['hybrid']).value
        L[3] += (logloss_sep(idx1[test_index], idx2[test_index], X_test, y_test, w1sep1.value, w1sep2.value)).value + lambd.value * (floss['individualsep']).value
        L[4] += (logloss_sep(idx1[test_index], idx2[test_index], X_test, y_test, w2sep1.value, w2sep2.value)).value + lambd.value * (floss['groupsep']).value
        L[5] += (logloss_sep(idx1[test_index], idx2[test_index], X_test, y_test, w3sep1.value, w3sep2.value)).value + lambd.value * (floss['hybridsep']).value
    return L


def get_loss_lin(X, y, idx1, idx2, g, ll):
    """
    Computes the total loss values for the given value of lambda in linear regression setting
    :param X: features
    :param y: ground-truth labels
    :param idx1: the indices of X that fall in protected group 1
    :param idx2: the indices of X that fall in protected group 2
    :param g: value of gamma
    :param ll: value of lambda
    :return:
    """

    L = [0, 0, 0, 0]
    lambd = cp.Parameter(nonneg=True)
    gamma = cp.Parameter(nonneg=True)
    # 10-fold cross validation for picking best gamma for given lambda
    cv = KFold(n_splits=10, shuffle=False)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        w1 = cp.Variable((X_train.shape[1], 1))
        w2 = cp.Variable((X_train.shape[1], 1))
        w1sep1 = cp.Variable((X_train.shape[1], 1))
        w1sep2 = cp.Variable((X_train.shape[1], 1))
        w2sep1 = cp.Variable((X_train.shape[1], 1))
        w2sep2 = cp.Variable((X_train.shape[1], 1))
        penal = fairness_penalty_lin(idx1[train_index], idx2[train_index], w1, w2, w1sep1, w1sep2, w2sep1, w2sep2, X_train, y_train)

        # minimize the objective function for individual fairness - single model
        problem_individual = cp.Problem(cp.Minimize(meanse(X_train, y_train, w1) + lambd * penal['individual'] + gamma * l2norm(w1)))
        lambd.value = ll
        gamma.value = g
        problem_individual.solve()

        # minimize the objective function for group fairness - single model
        problem_group = cp.Problem(cp.Minimize(meanse(X_train, y_train, w2) + lambd * penal['group'] + gamma * l2norm(w2)))
        lambd.value = ll
        gamma.value = g
        problem_group.solve()

        # minimize the objective function for individual fairness - separate model
        problem_individualsep = cp.Problem(cp.Minimize(meanse_sep(idx1[train_index], idx2[train_index], X_train, y_train, w1sep1, w1sep2) + lambd * penal['individualsep'] + gamma * (l2norm(w1sep1) + l2norm(w1sep2))))
        problem_individualsep.solve()

        # minimize the objective function for group fairness - separate model
        problem_groupsep = cp.Problem(cp.Minimize(meanse_sep(idx1[train_index], idx2[train_index], X_train, y_train, w2sep1, w2sep2) + lambd * penal['groupsep'] + gamma * (l2norm(w2sep1) + l2norm(w2sep2))))
        problem_groupsep.solve()

        # get loss values in test set
        floss = fairness_penalty_lin(idx1[test_index], idx2[test_index], w1.value, w2.value, w1sep1.value, w1sep2.value, w2sep1.value, w2sep2.value, X_test, y_test)
        L[0] += (meanse(X_test, y_test, w1.value)).value + lambd.value * (floss['individual']).value
        L[1] += (meanse(X_test, y_test, w2.value)).value + lambd.value * (floss['group']).value
        L[2] += (meanse_sep(idx1[test_index], idx2[test_index], X_test, y_test, w1sep1.value, w1sep2.value)).value + lambd.value * (floss['individualsep']).value
        L[3] += (meanse_sep(idx1[test_index], idx2[test_index], X_test, y_test, w2sep1.value, w2sep2.value)).value + lambd.value * (floss['groupsep']).value
    return L
