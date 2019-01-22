#!/usr/bin/env python3

import generate_input
import interleaving
import click_model as cm

import random

from scipy import stats, sqrt
from math import ceil



def tmp_interleaving(pair):
    """Temporary interleaving function for simulation test.
    
    Parameters
    ----------
    pair : tuple
        Tuple of ranking combinations.
    
    Returns
    -------
    out : list
        A list of interleaved ranking combinations.
    """
    out = []
    for item0, item1 in zip(pair[0], pair[1]):
        if random.random() < 0.5:
            out.append((0, item0))
            out.append((1, item1))
        else:
            out.append((1, item1))
            out.append((0, item0))
    return out

def tmp_click_model(search_results):
    """Temporary click model function for simulation test.
    
    Parameters
    ----------
    search_results : tuple
        Holds index of ranking algorithm of which ranking combination
        originates from and the ranking combination itself.
    
    Returns
    -------
    clicked_id : int
        Index of the ranking algorithm that is clicked.
    """
    index = len(search_results)
    while index >= len(search_results):
        index = int(abs(random.gauss(0, 1) / 2 * len(search_results)))
    clicked_id = search_results[index][0]
    return clicked_id


def interleaving_simulation(pair, k, interleaving_func, click_model):
    """Simulates user interaction on interleaved search results.
    
    Parameters
    ----------
    pair : tuple
        Pair of ranking combinations.
    k : int
        Number of simulations.
    interleaving_func : function(tuple) -> list
        Function to interleave the two lists of ranking combinations.
    click_model : function(array_like) -> int
        Function to simulate user click.
    
    Returns
    -------
    p : float
        Proportion of wins of second ranking combination in pair.
    """
    wins = [0] * len(pair)
    for _ in range(k):
        search_results = interleaving_func(pair)
        relevance = get_relevance(search_results)
        clicked = click_model(relevance)
        get_winner = winner(search_results,clicked)
        wins[winner] += 1
    p = wins[1] / float(wins[0] + wins[1])
    return p


def compute_sample_size(p1, alpha=0.05, beta=0.10):
    """Computes sample size for a given proportion
    based on power analysis. Returns -1 if p1 == 0.5.
    
    Parameters
    ----------
        p1 : float
            Proportion for which sample size is to be calculated.
        alpha : float
            Type I error parameter.
        beta : float
            Type II error parameter.
    
    Returns
    -------
        n : int
            Sample size.
    """
    p0 = 0.5
    # Return -1 if given proportion is exactly 50%.
    diff = p1 - p0
    if diff == 0:
        return -1
    # Compute sample size.
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(1 - beta)
    sigma0 = sqrt(p0 * (1 - p0))
    sigma1 = sqrt(p1 * (1 - p1))
    n = ((z_alpha * sigma0 + z_beta * sigma1) / diff) ** 2
    return ceil(n)


def process_bins(bins):
    """Prints minimum value, median and maximum value
    for each of the given bins.
    
    Parameters
    ----------
    bins : array_like
        Array of bins containing numerical values.
    
    Returns
    -------
    None
    """
    step = 1.0 / len(bins)
    for i, cur in enumerate(bins):
        # Determine delta range of bin.
        dmin = i * step
        dmax = dmin + step
        if i == 0:
            dmin += step / 2
        if i == len(bins) - 1:
            dmax -= step / 2
        dmin = round(dmin, 3)
        dmax = round(dmax, 3)
        range_str = '[' + str(dmin) + ' - ' + str(dmax) + ']'
        print('BIN', range_str)
        if len(cur) > 0:
            # Sort bin and remove error-data.
            cur.sort()
            cur.reverse()
            while cur[-1] == -1:
                cur.pop()
            cur.reverse()
            # Determine and print minimum, median and maximum.
            d, m = divmod(len(cur), 2)
            median = ceil((cur[d] + cur[-int(not bool(m))]) / 2.0)
            print('     min', min(cur))
            print('  median', median)
            print('     max', max(cur))
        else:
            print('  NO DATA')
        print()
    return


def main():
    inputs = generate_input.gen_input_pairs(3, 2)
    #Click model train

    bins_prob_random = [[] for _ in range(10)]
    bins_prob_pbm = [[] for _ in range(10)]
    bins_td_random = [[] for _ in range(10)]
    bins_td_pbm = [[] for _ in range(10)]
    bins = [bins_prob_random, bins_prob_pbm, bins_td_random, bins_td_pbm]
    length_interleaving = 3

    rcm = cm.RCM()
    pbm = cm.PBM()
    database = cm.read_yandex("./YandexRelPredChallenge.txt")
    rcm.learn(database)
    pbm.learn(database,3,5,length_interleaving)

    click_model_functions = [rcm.get_clicks,pbm.get_clicks]
    interleaving_functions = [interleaving.td_interleaving, interleaving.prob_interleaving]

    for pair in inputs:
        dERR = generate_input.ERR(pair[1]) - generate_input.ERR(pair[0])
        if dERR >= 0.05 and dERR <= 0.95:

            for permutation in generate_input.add_conflicts(pair): # Assumes (non-) duplicates for now.

                for i,click_model in enumerate(click_model_functions):
                    for j,interleaving in enumerate(interleaving_functions):
                        p = interleaving_simulation(
                            permutation, 100, interleaving,click_model)
                        bins[i*len(interleaving_functions)+j][int(dERR * 10)].append(compute_sample_size(p))

    process_bins(bins)
    return


if __name__ == '__main__':
    main()


