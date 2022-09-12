import numpy as np
import pandas as pd
import argparse
import os
import cvxpy as cp
import matplotlib.pyplot as plt
from multiprocessing import Pool
import itertools
import random
from src.loss import logloss, logloss_sep, meanse, meanse_sep
from src.fairness_penalty import fairness_penalty, fairness_penalty_lin
from CONSTANTS import PROCESSED_DATA_DIR, RAW_DATA_DIR, ROOT_DIR

random.seed(10)
np.random.seed(10)


def price(dataset, X, y, w, alpha, fp_wstar, fp, notion):
    """
    Computes the logloss/MSE loss in single model setting, when the predictor is constrained to have fairness penalty alpha
    :param dataset: name of the dataset
    :param X: features
    :param y: ground-truth labels
    :param w: regression weight vector
    :param alpha: alpha for PoF
    :param fp_wstar: fairness penalty of the unconstrained optimal predictor
    :param fp: dict of fairness penalties
    :param notion: notion of fairness
    :return: logloss/MSE loss in single model setting, when the predictor is constrained to have fairness penalty alpha
    """

    constraints = [fp[notion] <= (alpha * (fp_wstar[notion]).value)]
    if dataset == 'community':
        problem = cp.Problem(cp.Minimize(meanse(X, y, w)), constraints)
    else:
        problem = cp.Problem(cp.Minimize(logloss(X, y, w)), constraints)
    problem.solve(max_iters=500)
    return problem.value


def price_sep(dataset, X, y, idx1, idx2, w1, w2, alpha, fp_wstar, fp, notion):
    """
    Computes the logloss/MSE loss in separate model setting, when the predictor is constrained to have fairness penalty alpha
    :param dataset: name of the dataset
    :param X: features
    :param y: ground-truth labels
    :param idx1: the indices of X that fall in protected group 1
    :param idx2: the indices of X that fall in protected group 2
    :param w1: regression weight vector for group 1
    :param w2: regression weight vector for group 2
    :param alpha: alpha for PoF
    :param fp_wstar: fairness penalty of the unconstrained optimal predictor
    :param fp: dict of fairness penalties
    :param notion: notion of fairness
    :return: logloss/MSE loss in separate model setting, when the predictor is constrained to have fairness penalty alpha
    """

    constraints = [fp[notion] <= (alpha * (fp_wstar[notion]).value)]
    if dataset == 'community':
        problem = cp.Problem(cp.Minimize(meanse_sep(idx1, idx2, X, y, w1, w2)), constraints)
    else:
        problem = cp.Problem(cp.Minimize(logloss_sep(idx1, idx2, X, y, w1, w2)), constraints)
    problem.solve(max_iters=800)
    return problem.value


def main(X, y, idx1, idx2, dataset, alphas):
    # stack a column of 1s
    X = np.c_[np.ones(len(X)), X]

    w = cp.Variable((X.shape[1], 1))
    if dataset == 'community':
        problem = cp.Problem(cp.Minimize(meanse(X, y, w)))
        problem.solve()
    else:
        problem = cp.Problem(cp.Minimize(logloss(X, y, w)))
        problem.solve(max_iters=500)

    lp_wstar = problem.value
    if dataset == 'community':
        fp_wstar = fairness_penalty_lin(idx1, idx2, w, w, w, w, w, w, X, y, 0)
        pof = {'individual': [], 'group': [], 'individualsep': [], 'groupsep': []}
    else:
        fp_wstar = fairness_penalty(idx1, idx2, w, w, w, w, w, w, w, w, w, X, y, 0)
        pof = {'individual': [], 'group': [], 'hybrid': [], 'individualsep': [], 'groupsep': [], 'hybridsep': []}

    w1 = cp.Variable((X.shape[1], 1))
    w2 = cp.Variable((X.shape[1], 1))
    w3 = cp.Variable((X.shape[1], 1))
    w1sep1 = cp.Variable((X.shape[1], 1))
    w1sep2 = cp.Variable((X.shape[1], 1))
    w2sep1 = cp.Variable((X.shape[1], 1))
    w2sep2 = cp.Variable((X.shape[1], 1))
    w3sep1 = cp.Variable((X.shape[1], 1))
    w3sep2 = cp.Variable((X.shape[1], 1))
    random.seed(10)
    if dataset == 'community':
        fp = fairness_penalty_lin(idx1, idx2, w1, w2, w1sep1, w1sep2, w2sep1, w2sep2, X, y, 0)
    else:
        fp = fairness_penalty(idx1, idx2, w1, w2, w3, w1sep1, w1sep2, w2sep1, w2sep2, w3sep1, w3sep2, X, y, 0)

    pool = Pool(processes=proc)
    print("Processing individual-single")
    pof_values = pool.starmap(price, (itertools.product([dataset], [X], [y], [w1], alphas, [fp_wstar], [fp], ['individual'])))
    pool.close()
    pool.join()
    pof['individual'] = [x / lp_wstar for x in pof_values]

    pool = Pool(processes=proc)
    print("Processing group-single")
    pof_values = pool.starmap(price, (itertools.product([dataset], [X], [y], [w2], alphas, [fp_wstar], [fp], ['group'])))
    pool.close()
    pool.join()
    pof['group'] = [x / lp_wstar for x in pof_values]

    if dataset != 'community':
        pool = Pool(processes=proc)
        print("Processing hybrid-single")
        pof_values = pool.starmap(price, (itertools.product([dataset], [X], [y], [w3], alphas, [fp_wstar], [fp], ['hybrid'])))
        pool.close()
        pool.join()
        pof['hybrid'] = [x / lp_wstar for x in pof_values]

    pool = Pool(processes=proc)
    print("Processing individual-separate")
    pof_values = pool.starmap(price_sep, (itertools.product([dataset], [X], [y], [idx1], [idx2], [w1sep1], [w1sep2], alphas, [fp_wstar], [fp], ['individualsep'])))
    pool.close()
    pool.join()
    pof['individualsep'] = [x / lp_wstar for x in pof_values]

    pool = Pool(processes=proc)
    print("Processing group-separate")
    pof_values = pool.starmap(price_sep, (itertools.product([dataset], [X], [y], [idx1], [idx2], [w2sep1], [w2sep2], alphas, [fp_wstar], [fp], ['groupsep'])))
    pool.close()
    pool.join()
    pof['groupsep'] = [x / lp_wstar for x in pof_values]

    if dataset != 'community':
        pool = Pool(processes=proc)
        print("Processing hybrid-separate")
        pof_values = pool.starmap(price_sep, (itertools.product([dataset], [X], [y], [idx1], [idx2], [w3sep1], [w3sep2], alphas, [fp_wstar], [fp], ['hybridsep'])))
        pool.close()
        pool.join()
        pof['hybridsep'] = [x / lp_wstar for x in pof_values]

    print(pof)

    return pof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['compas', 'adult', 'community', 'default', 'lawschool'], required=True, help="dataset to use")
    parser.add_argument("--proc", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], default=7,
                        help="number of processes to run in parallel (between 1 and 16)")
    args = parser.parse_args()

    alphas = [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.04, 0.03, 0.02, 0.01]
    proc = args.proc

    data = None
    X = None
    y = None
    idx1 = None
    idx2 = None
    pof = dict()
    if args.dataset == 'compas':
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'COMPAS/compas_processed.csv'), index_col=None)
        y = data[['is_violent_recid']].values
        X = data.drop(['is_violent_recid', 'African-American', 'Caucasian'], axis=1).values
        # indices denoting which row is from which protected group
        idx1 = (data['African-American'] == 1).values
        idx2 = (data['Caucasian'] == 1).values
        pof = main(X, y, idx1, idx2, args.dataset, alphas)
    elif args.dataset == 'lawschool':
        data = pd.read_csv(os.path.join(RAW_DATA_DIR, 'Law School/lawschool.csv'), index_col=None)
        y = data[['bar1']].values
        X = data.drop(['bar1', 'gender'], axis=1).values
        # indices denoting which row is from which protected group
        idx1 = (data['gender'] == 1).values
        idx2 = (data['gender'] == 0).values
        pof = main(X, y, idx1, idx2, args.dataset, alphas)
    elif args.dataset == 'adult':
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'Adult/adult_processed.csv'), index_col=None)
        pof_temp = list()
        for i in range(3):
            print("Round " + str(i+1))
            data_sampled = data.sample(frac=0.3)
            y = data_sampled[['salary']].values
            X = data_sampled.drop(['salary', 'sex_Female', 'sex_Male'], axis=1).values
            # indices denoting which row is from which protected group
            idx1 = (data_sampled['sex_Male'] == 1).values
            idx2 = (data_sampled['sex_Female'] == 1).values
            pof_temp.append(main(X, y, idx1, idx2, args.dataset, alphas))
        for key in pof_temp[0].keys():
            # pof[key] = (pof_temp[0][key] + pof_temp[1][key] + pof_temp[2][key]) / 3
            pof[key] = [sum(x)/3 for x in zip(pof_temp[0][key], pof_temp[1][key], pof_temp[2][key])]
    elif args.dataset == 'default':
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'Default/default_processed.csv'), index_col=None)
        pof_temp = list()
        for i in range(3):
            print("Round " + str(i + 1))
            data_sampled = data.sample(frac=0.3)
            y = data_sampled[['default payment next month']].values
            X = data_sampled.drop(['default payment next month', 'SEX'], axis=1).values
            # indices denoting which row is from which protected group
            idx1 = (data_sampled['SEX'] == 1).values
            idx2 = (data_sampled['SEX'] == 2).values
            pof_temp.append(main(X, y, idx1, idx2, args.dataset, alphas))
        for key in pof_temp[0].keys():
            pof[key] = [sum(x) / 3 for x in zip(pof_temp[0][key], pof_temp[1][key], pof_temp[2][key])]
    elif args.dataset == 'community':
        data = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'Communities and Crime/community_processed.csv'), index_col=None)
        # indices denoting which row is from which protected group
        idx1 = ((data['blackPerCap'] >= data['whitePerCap']) & (data['blackPerCap'] >= data['AsianPerCap']) & (data['blackPerCap'] >= data['indianPerCap']) & (data['blackPerCap'] >= data['HispPerCap'])).values
        idx2 = np.invert(idx1)
        data = (data - data.mean()) / data.std()
        y = data[['ViolentCrimesPerPop']].values
        X = data.drop(['ViolentCrimesPerPop'], axis=1).values

        pof = main(X, y, idx1, idx2, args.dataset, alphas)

    x = np.arange(len(alphas))  # the label locations
    width = 0.1  # the width of the bars
    fig, ax = plt.subplots()

    if args.dataset == 'community':
        _ = ax.bar(x - 2 * width, pof['groupsep'], width, label='Group, separate', color='b')
        _ = ax.bar(x - width, pof['group'], width, label='Group, single', color='r')
        _ = ax.bar(x, pof['individualsep'], width, label='Individual, separate', color='c')
        _ = ax.bar(x + width, pof['individual'], width, label='Individual, single', color='g')
    else:
        _ = ax.bar(x - 2 * width, pof['groupsep'], width, label='Group, separate', color='b')
        _ = ax.bar(x - width, pof['group'], width, label='Group, single', color='r')
        _ = ax.bar(x, pof['hybridsep'], width, label='Hybrid, separate', color='m')
        _ = ax.bar(x + width, pof['hybrid'], width, label='Hybrid, single', color='y')
        _ = ax.bar(x + 2 * width, pof['individualsep'], width, label='Individual, separate', color='c')
        _ = ax.bar(x + 3 * width, pof['individual'], width, label='Individual, single', color='g')

    ax.set_ylabel('Price of Fairness')
    ax.set_xlabel('alpha')
    ax.set_title(args.dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(alphas)
    ax.set_ylim([0.5, 3])
    ax.legend(loc='upper left')
    fig.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(ROOT_DIR, "output", args.dataset + "_pof.png"))
