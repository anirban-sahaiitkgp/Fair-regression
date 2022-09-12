import numpy as np
import cvxpy as cp
import random
import itertools

random.seed(10)
np.random.seed(10)


def fairness_penalty(indices1, indices2, w1, w2, w3, w1sep1, w1sep2, w2sep1, w2sep2, w3sep1, w3sep2, X, y, flg=0):
    """
    Computes fairness penalties for logistic regression setting
    :param indices1: the indices of X that fall in protected group 1
    :param indices2: the indices of X that fall in protected group 2
    :param w1: regression weights for individual fairness - single model
    :param w2: regression weights for group fairness - single model
    :param w3: regression weights for hybrid fairness - single model
    :param w1sep1: regression weights for individual fairness for group 1 - separate model
    :param w1sep2: regression weights for individual fairness for group 2 - separate model
    :param w2sep1: regression weights for group fairness for group 1 - separate model
    :param w2sep2: regression weights for group fairness for group 2 - separate model
    :param w3sep1: regression weights for hybrid fairness for group 1 - separate model
    :param w3sep2: regression weights for hybrid fairness for group 2 - separate model
    :param X: features
    :param y: ground-truth labels
    :param flg: if 1, calculate the actual fairness loss in the test set. Otherwise, return the cvxpy object
    :return: fairness penalty values or corresponding cvxpy object
    """

    if not flg:
        n1 = np.sum(indices1)
        n2 = np.sum(indices2)
        n = min(n1, n2)
        i1 = np.where(indices1)[0]
        i2 = np.where(indices2)[0]
        # randomly sample 2n cross-pairs
        while True:
            temp = np.array(random.sample(list(itertools.product(i1, i2)), k=2*n))
            smp1 = temp[:, 0]
            smp2 = temp[:, 1]
            S1 = X[smp1]
            y1 = y[smp1]
            S2 = X[smp2]
            y2 = y[smp2]
            n_0 = np.sum(np.logical_and(y1 == 0, y2 == 0))
            n_1 = np.sum(np.logical_and(y1 == 1, y2 == 1))
            if n_0 != 0 and n_1 != 0:
                break

        penal = {'individual': 0, 'group': 0, 'hybrid': 0, 'individualsep': 0, 'groupsep': 0, 'hybridsep': 0}

        # individual fairness penalty - single model
        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w1 - S2 @ w1)**2))
        penal['individual'] = summation / (2*n)

        # group fairness penalty - single model
        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w2 - S2 @ w2)))
        penal['group'] = (summation / (2*n))**2

        # hybrid fairness penalty - single model
        summation = ((1 / n_0) * cp.sum(cp.multiply(cp.multiply((y1 == 0), (y2 == 0)), (S1 @ w3 - S2 @ w3))))**2
        summation += ((1 / n_1) * cp.sum(cp.multiply(cp.multiply((y1 == 1), (y2 == 1)), (S1 @ w3 - S2 @ w3))))**2
        penal['hybrid'] = summation

        # individual fairness penalty - separate model
        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w1sep1 - S2 @ w1sep2) ** 2))
        penal['individualsep'] = summation / (2 * n)

        # group fairness penalty - separate model
        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w2sep1 - S2 @ w2sep2)))
        penal['groupsep'] = (summation / (2 * n)) ** 2

        # hybrid fairness penalty - separate model
        summation = ((1 / n_0) * cp.sum(cp.multiply(cp.multiply((y1 == 0), (y2 == 0)), (S1 @ w3sep1 - S2 @ w3sep2)))) ** 2
        summation += ((1 / n_1) * cp.sum(cp.multiply(cp.multiply((y1 == 1), (y2 == 1)), (S1 @ w3sep1 - S2 @ w3sep2)))) ** 2
        penal['hybridsep'] = summation

        return penal
    else:
        n1 = np.sum(indices1)
        n2 = np.sum(indices2)
        i1 = np.where(indices1)[0]
        i2 = np.where(indices2)[0]

        # create all possible cross-pairs
        temp = np.array(random.sample(list(itertools.product(i1, i2)), k=n1*n2))
        smp1 = temp[:, 0]
        smp2 = temp[:, 1]
        S1 = X[smp1]
        y1 = y[smp1]
        S2 = X[smp2]
        y2 = y[smp2]
        penal = {'individual': 0, 'group': 0, 'hybrid': 0, 'individualsep': 0, 'groupsep': 0, 'hybridsep': 0}

        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w1 - S2 @ w1)**2))
        penal['individual'] = summation / (n1*n2)

        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w2 - S2 @ w2)))
        penal['group'] = (summation / (n1*n2))**2
        
        n10 = np.sum(y[i1] == 0)
        n11 = np.sum(y[i1] == 1)
        n20 = np.sum(y[i2] == 0)
        n21 = np.sum(y[i2] == 1)
        summation = ((1 / (n10 * n20)) * cp.sum(cp.multiply(cp.multiply((y1 == 0), (y2 == 0)), (S1 @ w3 - S2 @ w3))))**2
        summation += ((1 / (n11 * n21)) * cp.sum(cp.multiply(cp.multiply((y1 == 1), (y2 == 1)), (S1 @ w3 - S2 @ w3))))**2
        penal['hybrid'] = summation

        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w1sep1 - S2 @ w1sep2) ** 2))
        penal['individualsep'] = summation / (n1 * n2)

        summation = cp.sum(cp.multiply((y1 == y2), (S1 @ w2sep1 - S2 @ w2sep2)))
        penal['groupsep'] = (summation / (n1 * n2)) ** 2

        summation = ((1 / (n10 * n20)) * cp.sum(cp.multiply(cp.multiply((y1 == 0), (y2 == 0)), (S1 @ w3sep1 - S2 @ w3sep2)))) ** 2
        summation += ((1 / (n11 * n21)) * cp.sum(cp.multiply(cp.multiply((y1 == 1), (y2 == 1)), (S1 @ w3sep1 - S2 @ w3sep2)))) ** 2
        penal['hybridsep'] = summation
        
        return penal


def fairness_penalty_lin(indices1, indices2, w1, w2, w1sep1, w1sep2, w2sep1, w2sep2, X, y, flg=0):
    """
        Computes fairness penalties for linear regression setting
        :param indices1: the indices of X that fall in protected group 1
        :param indices2: the indices of X that fall in protected group 2
        :param w1: regression weights for individual fairness - single model
        :param w2: regression weights for group fairness - single model
        :param w1sep1: regression weights for individual fairness for group 1 - separate model
        :param w1sep2: regression weights for individual fairness for group 2 - separate model
        :param w2sep1: regression weights for group fairness for group 1 - separate model
        :param w2sep2: regression weights for group fairness for group 2 - separate model
        :param X: features
        :param y: ground-truth labels
        :param flg: if 1, calculate the actual fairness loss in the test set. Otherwise, return the cvxpy object
        :return: fairness penalty values or corresponding cvxpy object
        """

    if not flg:
        n1 = np.sum(indices1)
        n2 = np.sum(indices2)
        n = min(n1, n2)
        i1 = np.where(indices1)[0]
        i2 = np.where(indices2)[0]
        # randomly sample 2n cross-pairs
        temp = np.array(random.sample(list(itertools.product(i1, i2)), k=2 * n))
        smp1 = temp[:, 0]
        smp2 = temp[:, 1]
        S1 = X[smp1]
        y1 = y[smp1]
        S2 = X[smp2]
        y2 = y[smp2]

        penal = {'individual': 0, 'group': 0, 'individualsep': 0, 'groupsep': 0}

        # individual fairness penalty - single model
        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w1 - S2 @ w1) ** 2))
        penal['individual'] = summation / (2 * n)

        # group fairness penalty - single model
        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w2 - S2 @ w2)))
        penal['group'] = (summation / (2 * n)) ** 2

        # individual fairness penalty - separate model
        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w1sep1 - S2 @ w1sep2) ** 2))
        penal['individualsep'] = summation / (2 * n)

        # group fairness penalty - separate model
        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w2sep1 - S2 @ w2sep2)))
        penal['groupsep'] = (summation / (2 * n)) ** 2

        return penal
    else:
        n1 = np.sum(indices1)
        n2 = np.sum(indices2)
        i1 = np.where(indices1)[0]
        i2 = np.where(indices2)[0]

        # create all possible cross-pairs
        temp = np.array(random.sample(list(itertools.product(i1, i2)), k=n1 * n2))
        smp1 = temp[:, 0]
        smp2 = temp[:, 1]
        S1 = X[smp1]
        y1 = y[smp1]
        S2 = X[smp2]
        y2 = y[smp2]
        penal = {'individual': 0, 'group': 0, 'individualsep': 0, 'groupsep': 0}

        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w1 - S2 @ w1) ** 2))
        penal['individual'] = summation / (n1 * n2)

        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w2 - S2 @ w2)))
        penal['group'] = (summation / (n1 * n2)) ** 2

        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w1sep1 - S2 @ w1sep2) ** 2))
        penal['individualsep'] = summation / (n1 * n2)

        summation = cp.sum(cp.multiply(cp.exp(-(y1 - y2)**2), (S1 @ w2sep1 - S2 @ w2sep2)))
        penal['groupsep'] = (summation / (n1 * n2)) ** 2

        return penal
