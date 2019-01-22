#!/usr/bin/env python3

import generate_input
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
    out : list
        A list containing the index of the document that is clicked.
    """
    index = len(search_results)
    while index >= len(search_results):
        index = int(abs(random.gauss(0, 1) / 2 * len(search_results)))
    clicked_id = search_results[index][0]
    out = [clicked_id]
    return out


def interleaving_simulation(pair, k, interleaving_func, click_model_func, length_interleaving=3):
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
        search_results = interleaving_func(pair, length_interleaving)  # Create interleaved list

        relevance_grades = []  # Get relevance label of documents that were in the interleavedl ist
        for relevance, assignment in search_results:
            relevance_grades.append(relevance)

        clicked = click_model_func(relevance_grades)  # Get clicked documents indices from interleaved list
        counter_E_click = 0
        counter_P_click = 0
        for click in clicked:  # Determine who got most clicks
            assignment = search_results[click][1]
            if assignment == 1:
                counter_E_click += 1
            else:
                counter_P_click += 1

        if counter_E_click == counter_P_click:
            continue
        elif counter_E_click > (length_interleaving / 2):
            winner = 1
        else:
            winner = 0

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
    bins = [[] for _ in range(10)]
    for pair in inputs:
        dERR = generate_input.ERR(pair[1]) - generate_input.ERR(pair[0])
        if dERR >= 0.05 and dERR <= 0.95:
            # Assumes no duplicates for now.
            p = interleaving_simulation(
                pair, 1000, tmp_interleaving, tmp_click_model)
            bins[int(dERR * 10)].append(compute_sample_size(p))
    process_bins(bins)
    return


if __name__ == '__main__':
    main()


