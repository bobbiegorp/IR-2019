import numpy as np


def td_interleaving(ranking_pair,max_interleav=3):
    """Run Team-draft interleaving given a ranking pair as input

    Parameters
    ----------
    An ranking pair : List of ranked and labeled results
        Index of list elements represents the rank of that element. Each
        element is a tuple of the relevance score and a duplicate ID.
        If this value is 0, it has no duplicate with a document in the results of other ranker.
        If the value is greater than 0, then it has a duplicate with another result of the other ranker that matches this number
        No duplicate example: E ranked list:  [(0,0),(0,0),(0,0)] and P ranked list: [(1,0),(0,0),(0,0)] form
        2 duplicates example: E ranked list: [(1,1),(0,2),(0,0)]  and P ranked list: [(1,0),(1,1),(0,2)]  for example
    Returns
    -------
    Interleaved list (of length 3 as default) based on Team-draft method: list
        Index + 1 represents the rank of the interleaved list and element is an tuple of the form (relevance: binary,ranker credit:binary), Credits are assigned as P(0) and E(1)
    """
    ranking_p = ranking_pair[0] #[(0,0),(0,0),(0,0)] form or duplicate [(0,1),(0,0),(0,0)]
    ranking_e = ranking_pair[1]
    interleaved = []

    p_team = 0 #Amount results assigned from p
    e_team = 0
    p_pointer = 0 #Next top result from ranking p
    e_pointer = 0
    found_duplicates = [] #Duplicate documents have an ID of greater than 0. A matching number is an duplciate
    limit = len(ranking_p)

    #while p_pointer < limit and e_pointer < limit:
    while len(interleaved) < max_interleav:

        p_priority = np.random.choice(2, 1)[0]
        new_result = False
        if (p_team < e_team) or (p_team == e_team and p_priority == 1):

            while not new_result:
                top_result_p = ranking_p[p_pointer]
                relevance_p, duplicate_id_p = top_result_p

                p_pointer += 1
                if duplicate_id_p not in found_duplicates:
                    new_result = True
                    break
                elif p_pointer == limit:
                    break

            if new_result:
                #interleaved.append( (relevance_p,"P") )
                interleaved.append((relevance_p, 0))
                p_team += 1

                if duplicate_id_p > 0:
                    found_duplicates.append(duplicate_id_p)
        else:

            while not new_result:
                top_result_e = ranking_e[e_pointer]
                relevance_e, duplicate_id_e = top_result_e
                e_pointer += 1

                if duplicate_id_e not in found_duplicates:
                    new_result = True
                    break
                elif e_pointer == limit:
                    break

            if new_result:
                #interleaved.append((relevance_e, "E"))
                interleaved.append((relevance_e, 1))
                e_team += 1

                if duplicate_id_e > 0:
                    found_duplicates.append(duplicate_id_e)

    return interleaved

def get_softmax(ranking_indices,tau):
    """Compute softmax distribution for a ranker given the indices of documents that are avaliable to be picked.

    Parameters
    ----------
    Ranking indices : List of indices that represent the rank of documents that can be picked for next interleaving.
    Example of nothing picked of ranker: List of the form: [0,1,2]
    Example of document already picked of ranker: List of the form: [0,2]

    Returns
    -------
    List of softmax probabilities for documents to be picked: List
        The list maps one to one the ranking indices that were given as input, each element is a probability for the document for that particular index
        in that ranking indices list, which in turn consist of numbers that represents the actual indices to be picked from ranked results.

    """

    numerator_list = [] #Numerator values for each of the ranked results
    softmax_distribution = []

    for rank_index in ranking_indices:
        rank = rank_index + 1
        numerator_value = 1/(rank**tau)
        numerator_list.append(numerator_value)

    denominator = sum(numerator_list)

    for value in numerator_list:
        probability = value/denominator
        softmax_distribution.append(probability)

    return softmax_distribution

def prob_interleaving(ranking_pair,max_interleav=3,tau=3):
    """Run Probabilistic interleaving  given a ranking pair as input

    Parameters
    ----------
    An ranking pair : List of ranked and labeled results
        Index of list elements represents the rank of that element. Each
        element is a tuple of the relevance score and a duplicate ID.
        If this value is 0, it has no duplicate with a document in the results of other ranker.
        If the value is greater than 0, then it has a duplicate with another result of the other ranker that matches this number
        No duplicate example: E ranked list:  [(0,0),(0,0),(0,0)] and P ranked list: [(1,0),(0,0),(0,0)] form
        2 duplicates example: E ranked list: [(1,1),(0,2),(0,0)]  and P ranked list: [(1,0),(1,1),(0,2)]  for example
    Returns
    -------
    Interleaved list (of length 3 as default) based on probabilistc interleaving method: list
        Index + 1 represents the rank of the interleaved list and element is an tuple of the form (relevance: binary,ranker credit:binary), credits are assigned as P(0) and E(1)
    """

    ranking_p = ranking_pair[0] #[(0,0),(0,0),(0,0)] form or duplicate [(0,1),(0,0),(0,0)]
    ranking_e = ranking_pair[1] #[(0,0),(0,0),(0,0)] form or duplicate #[(0,1),(0,0),(0,0)]
    interleaved = []
    limit = len(ranking_p)

    p_indices = list(range(limit))
    e_indices = list(range(limit))

    found_duplicates = []

    #while len(p_indices) > 0 or len(e_indices) > 0:
    while len(interleaved) < max_interleav:

        p_priority = np.random.choice(2, 1)[0]

        if (p_priority and len(p_indices) > 0) or len(e_indices) == 0:
            softmax_p = get_softmax(p_indices,tau)
            doc_index_p = np.random.choice(p_indices, 1, p=softmax_p)[0]
            p_indices.remove(doc_index_p)

            result_p = ranking_p[doc_index_p]
            relevance_p, duplicate_id_p = result_p

            if duplicate_id_p == 0 :
                #interleaved.append((relevance_p, "P"))
                interleaved.append((relevance_p, 0))
            elif (duplicate_id_p > 0 and duplicate_id_p not in found_duplicates):
                #interleaved.append((relevance_p, "P"))
                interleaved.append((relevance_p, 0))
                found_duplicates.append(duplicate_id_p)
                duplicate_index = ranking_e.index(result_p)
                e_indices.remove(duplicate_index)
        else:
            softmax_e = get_softmax(e_indices,tau)
            doc_index_e = np.random.choice(e_indices, 1, p=softmax_e)[0]
            e_indices.remove(doc_index_e)

            result_e = ranking_e[doc_index_e]
            relevance_e, duplicate_id_e = result_e

            if duplicate_id_e == 0:
                #interleaved.append((relevance_e, "E"))
                interleaved.append((relevance_e, 1))
            elif (duplicate_id_e > 0 and duplicate_id_e not in found_duplicates):
                #interleaved.append((relevance_e, "E"))
                interleaved.append((relevance_e, 1))
                found_duplicates.append(duplicate_id_e)
                duplicate_index = ranking_p.index(result_e)
                p_indices.remove(duplicate_index)

    return interleaved

def main():
    pair = [[(0,1),(0,2),(0,3),(0,4)],[(0,2),(0,3),(0,4),(0,1)]]
    print(pair[0],"\n")
    print(pair[1],"\n")
    for i in range(5):
        interleav = td_interleaving(pair)
        print("Teamdraft: ", interleav)

    for i in range(5):
        interleav = prob_interleaving(pair)
        print("Probabilistic:",interleav)

    print("---------------------Softmax ------------\n")
    softmax = get_softmax(list(range(3)),3)
    print(softmax)

    print("---------------------New Pair ------------\n")
    pair = [[(0,1),(1,2),(0,3)],[(1,2),(0,1),(0,3)]]
    print(pair[0],"\n")
    print(pair[1],"\n")

    for i in range(5):
        interleav = prob_interleaving(pair)
        print("Probabilistic:",interleav)

    print("---------------------New Pair ------------\n")

    pair = [[(0,0),(1,0),(0,0)],[(1,0),(0,0),(0,0)]]
    print(pair[0],"\n")
    print(pair[1],"\n")

    for i in range(5):
        interleav = prob_interleaving(pair)
        print("Probabilistic:",interleav)

if __name__ == '__main__':
    main()