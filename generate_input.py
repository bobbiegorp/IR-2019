#!/usr/bin/env python3


def gen_input_unsorted(length, n):
    """Creates a list of all possible combinations of relevance scores.
    
    Parameters
    ----------
    length : int
        Length of a combination of relevance scores.
    n : int
        Maximum relevance score
    
    Returns
    -------
    out : list
        A list containing all possible combinations of relevance scores.
    """
    out = []
    out.append([0] * length)
    for i in range(length):
        for j in range(1, n):
            affix = [0] * i + [j]
            suffixes = gen_input_unsorted(length - (i + 1), n)
            for suffix in suffixes:
                out.append(affix + suffix)
    return out

def gen_input(length, n):
    """Creates a sorted list of all possible combinations
    of relevance scores.
    
    Parameters
    ----------
    length : int
        Length of a combination of relevance scores.
    n : int
        Maximum relevance score
    
    Returns
    -------
    out : list
        A sorted list containing all possible combinations
        of relevance scores.
    """
    if length > 0:
        out = gen_input_unsorted(length, n)
        out.sort()
    else:
        out = []
    return out

def gen_input_pairs(length, n):
    """Creates a sorted list of all possible pairs of combinations
    of relevance scores.
    
    Parameters
    ----------
    length : int
        Length of a combination of relevance scores.
    n : int
        Maximum relevance score
    
    Returns
    -------
    out : list
        A sorted list containing all possible pairs of combinations
        of relevance scores.
    """
    out = gen_input(length * 2, n)
    for i in range(len(out)):
        out[i] = (out[i][:length], out[i][length:])
    return out


def get_conflicts(n, length, _in=[], ordered=False):
    """Creates a list of all possible id conflicts.
    
    Parameters
    ----------
    n : int
        Total possible number of id conflicts at a time.
    length: int
        Length of the list containing id conflicts.
    _in : array_like
        Array of current id conflicts.
    ordered : bool
        Indicates whether id conflict numbers are to appear in order
        or are permitted to appear in any order.
    
    Returns
    -------
    out : list
        A list containing all possible id conflicts.
        A value of 0 indicates no conflict,
        a value higher than indicates a conflict.
    """
    if (len(_in) < length and n <= length):
        out = []
        # Add all possible conflicts.
        for i in [x for x in range(1, n + 1) if x not in _in]:
            out += get_conflicts(n, length, _in + [i], ordered)
            if ordered:
                break
        # Add absence of a conflict if possible.
        if (length - len(_in) > n - sum([1 for i in _in if i != 0])):
            out += get_conflicts(n, length, _in + [0], ordered)
        return out
    else:
        return [_in]

def add_conflicts(pair):
    """Adds id conflicts to a tuple of combinations of relevance grades.
    
    Parameters
    ----------
    pair : tuple
        Tuple of combinations of relevance grades.
    
    Returns
    -------
    out : list
        A list of all possible tuples of combinations
        of relevance grades with id conflicts.
        Relevance grades from the inputs have been replaced by a tuple
        of relevance grade and id conflict number.
    """
    out = []
    # Add all possible id conflicts to input pair.
    for n in range(len(pair[0]) + 1):
        for ids0 in get_conflicts(n, len(pair[0]), ordered=True):
            ranking0 = list(zip(pair[0], ids0))
            for ids1 in get_conflicts(n, len(pair[0])):
                out.append((ranking0, list(zip(pair[1], ids1))))
    # Remove any pairs in which conflicts appear
    # where relevance grades do not match.
    for i in range(len(out) - 1, -1, -1):
        delete = False
        for r0, id0 in out[i][0]:
            if id0 > 0:
                for r1, id1 in out[i][1]:
                    if id0 == id1 and not r0 == r1:
                        delete = True
                        break
            if delete:
                out.pop(i)
                break
    return out


def ERR(g_list, R_func=lambda g, max_g: float(2**g- 1) / 2**max_g):
    """Calculates Expected Reciprocal Rank for one list
    of relevance grades.
    
    Source
    ------
    The algorithm originates from a paper by O. Chapelle et al.
    named "Expected Reciprocal Rank for Graded Relevance"
    and can be found here: http://olivier.chapelle.cc/pub/err.pdf
    
    Parameters
    ----------
    g_list : array_like
        Array of relevance grades.
    R_func : function(g, max_g)
        Function that converts a relevance grade
        to probability of relevance.
    
    Returns
    -------
    ERR : float
        Expected Reciprocal Rank
    """
    p = 1
    ERR = 0
    max_g = max(g_list)
    for r in range(1, len(g_list) + 1):
        R = R_func(g_list[r - 1], max_g)
        ERR += p * R / float(r)
        p *= 1 - R
    return ERR


def main():
    inputs = gen_input_pairs(3, 2)
    for _in in inputs:
        print('\nCurrent input:', _in)
        print('    DELTA ERR:', ERR(_in[1]) - ERR(_in[0]))
        print('\nPossible conflicts\n------------------')
        for pair in add_conflicts(_in):
            print(pair)
        input('Press enter to continue\n')
    return


if __name__ == '__main__':
    main()


