#!/usr/bin/env python3

import generate_input
import random

from scipy import stats, sqrt
from math import ceil

import matplotlib.pyplot as plt

def tmp_interleaving(pair, length=-1):
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
    if length < 0:
        length = sum([len(p) for p in pair])
    for item0, item1 in zip(pair[0], pair[1]):
        if random.random() < 0.5:
            out.append((item0, 0))
            out.append((item1, 1))
        else:
            out.append((item1, 1))
            out.append((item0, 0))
    return out[:length]

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
    out = [index]
    return out


def interleaving_simulation(pair, k, interleaving_func, click_model_func, length_interleaving=-1):
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
        # Create interleaved list
        search_results = interleaving_func(pair, length_interleaving)
        # Get relevance label of documents in interleaved list
        relevance_grades = []
        for relevance, assignment in search_results:
            relevance_grades.append(relevance)
        # Get clicked documents indices from interleaved list
        clicked = click_model_func(relevance_grades)
        n_E_click = 0
        n_P_click = 0
        # Determine who got most clicks
        for click in clicked:
            assignment = search_results[click][1]
            if assignment == 1:
                n_E_click += 1
            else:
                n_P_click += 1

        if n_E_click == n_P_click:
            wins[0] += 1
            wins[1] += 1
        elif n_E_click > n_P_click:
            wins[1] += 1
        else:
            wins[0] += 1

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


def get_bin_labels(n_bins, n_decimals=3, cut_sides=0.0):
    """Creates labels containing ranges for bins.
    
    Parameters
    ----------
    n_bins : int
        Amount of bins to label.
    n_decimals : int
        Number of decimals to round the ranges on.
    cut_sides : float
        Amount by which first and last bin have their ranges cut.
    
    Returns
    -------
    out : list
        A list containing labels.
    """
    out = []
    step = 1.0 / n_bins
    for i in range(n_bins):
        # Determine delta range of bin.
        dmin = i * step
        dmax = dmin + step
        if i == 0:
            dmin += cut_sides
        if i == n_bins - 1:
            dmax -= cut_sides
        dmin = round(dmin, n_decimals)
        dmax = round(dmax, n_decimals)
        out.append('[' + str(dmin) + ' - ' + str(dmax) + ')')
    return out

def process_bins(bins):
    """Determines minimum value, maximum value and median
    for each of the given bins.
    
    Parameters
    ----------
    bins : array_like
        Array of bins containing numerical values.
    
    Returns
    -------
    out : list
        A list of dictionaries containing the minimum value,
        maximum value and median for each bin.
    """
    out = []
    for cur in bins:
        if cur == []:
            out.append({'has_info' : False})
        else:
            # Sort bin and remove error-data.
            cur.sort()
            cur.reverse()
            while cur[-1] == -1:
                cur.pop()
            cur.reverse()
            # Determine minimum, median and maximum.
            d, m = divmod(len(cur), 2)
            median = ceil((cur[d] + cur[-int(not bool(m))]) / 2.0)
            out.append({'min' : min(cur), 'max' : max(cur),
                'median' : median, 'has_info' : True})
    return out

def print_bin_info(bin_info, labels=None):
    """Prints minimum value, median and maximum value
    for each of the given bin info dictionaries.
    
    Parameters
    ----------
    bins : array_like
        Array of dictionaries containing `min`, `max` and `median`
        fields, also contains a `has_info` field which is to be turned
        set `False` if not all other fields are present.
        
    Returns
    -------
    None
    """
    if labels == None:
        labels = list(range(len(bin_info)))
    for info, label in zip(bin_info, labels):
        print('BIN', label)
        if info['has_info']:
            print('     min', info['min'])
            print('  median', info['median'])
            print('     max', info['max'])
        else:
            print('  NO DATA')
        print()
    return


def plot_bin_info(bin_info_list, bin_set_labels=[], bin_labels=[]):
    """Plots min/median/max information for the bins
    in the given bin sets.
    
    Parameters
    ----------
    bin_info_list : array_like
        A list of sets of bins containing dictionaries
        with min/median/max/has_info information.
    bin_set_labels : array_like
        A list of labels for the different sets of bins.
    bin_labels : array_like
        A list of labels of the different bins in the bin sets.
        Is used to label the x axis.
    
    Returns
    -------
    None
    """
    if bin_set_labels == []:
        bin_set_labels = list(range(len(bin_info_list)))
    # Plot median with min and max as error bars
    # for each bin in each bin set.
    for i, (bin_info, label) in enumerate(zip(bin_info_list, bin_set_labels)):
        x = [j + 0.1 * (i + 1) for j in range(len(bin_info))]
        y = [info['median'] if info['has_info'] else 0 for info in bin_info]
        err = ([info['median'] - info['min'] if info['has_info'] else 0
            for info in bin_info],
            [info['max'] - info['median'] if info['has_info'] else 0
            for info in bin_info])
        plt.errorbar(x, y, err, label=label)
    # Apply bin labels.
    if bin_labels != []:
        plt.xticks(range(len(bin_labels)), bin_labels + [''], rotation=30)
    plt.legend()
    plt.xlabel('$\Delta$ERR')
    plt.ylabel('sample size')
    plt.title('Determined sample size')
    plt.tight_layout()
    plt.show()
    return


def main():
    n_bins = 10
    cut_sides = 0.05
    inputs = generate_input.gen_input_pairs(3, 2)
    bins = [[[] for _ in range(n_bins)] for _ in range(2)]
    # Obtain data through simulations.
    for pair in inputs:
        dERR = generate_input.ERR(pair[1]) - generate_input.ERR(pair[0])
        if dERR >= cut_sides and dERR < 1.0 - cut_sides:
            # Assumes no duplicates for now.
            for cur_bins in bins:
                p = interleaving_simulation(
                    pair, 50000, tmp_interleaving, tmp_click_model)
                cur_bins[int(dERR * 10)].append(compute_sample_size(p))
    # Process bins.
    bin_labels = get_bin_labels(n_bins, cut_sides=cut_sides)
    bin_info_list = []
    bin_set_labels = [str(i) for i in range(len(bin_info_list))]
    for i, cur_bins in enumerate(bins):
        print('RUN ' + str(i) + '\n=====\n')
        bin_info =  process_bins(cur_bins)
        print_bin_info(bin_info, bin_labels)
        bin_info_list.append(bin_info)
    # Plot bin information.
    plot_bin_info(bin_info_list, bin_set_labels, bin_labels)
    return


if __name__ == '__main__':
    main()


