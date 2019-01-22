#!/usr/bin/env python3

from random import random
import matplotlib.pyplot as plt


def read_yandex(path, n=-1):
    """Reads yandex database.
    
    Parameters
    ----------
    path : str
        Path to database file.
    n : int
        Number of lines to read. Is ignored if value is lower than 0.
    
    Returns
    -------
    out : list
        A list of dictionaries representing database entries.
    """
    out = []
    with open(path) as f:
        for i, l in enumerate(f):
            if n >= 0 and i > n:
                break
            data = l.strip().split('\t')
            item = {
                'id'   : int(data[0]),
                't'    : int(data[1]),
                'a'    : data[2].lower(),
                'a_id' : int(data[3])
            }
            if item['a'] == 'q':
                item['r_id'] = int(data[4])
                item['urls'] = [int(x) for x in data[5:]]
            out.append(item)
    return out


class RCM:
    """Random Click Model
    ==================
    
    Click model that simulates user interaction as random clicking.
    
    One parameter `rho` is learned signifying the chance
    of a document being clicked. User interaction is then simulated
    by comparing a random value drawn from a uniform distribution
    against this `rho` value.
    """
    def __init__(self):
        """Initializes class parameters.
        """
        self.rho = random()
    
    def learn(self, database):
        """Learns class parameters.
        
        Parameters
        ----------
        database : array_like
            Array of dictionaries representing database items.
        
        Returns
        -------
        None
        """
        n_clicks, n_docs = 0, 0
        for item in database:
            if item['a'] == 'q':
                n_docs += len(item['urls'])
            else:
                n_clicks += 1
        self.rho = n_clicks / float(n_docs)
        return
    
    def get_p(self, relevance_grades):
        """Determines chance of clicking on a document.
        
        Parameters
        ----------
        relevance_grades : array_like
            Array containing relevance grades for all documents
            returned by a search query.
        
        Returns
        -------
        out : list
            List of probabilities corresponding
            to entries in `search_results`.
        """
        out = [self.rho for _ in range(len(relevance_grades))]
        return out
    
    def get_clicks(self, relevance_grades):
        """Simulate user interaction by determining
        what documents are clicked on.
        
        Parameters
        ----------
        relevance_grades : array_like
            Array containing relevance grades for all documents
            returned by a search query.
        
        Returns
        -------
        out : list
            List of indices of the documents that were clicked on
            in the simulation.
        """
        p = self.get_p(relevance_grades)
        out = []
        for i in range(len(p)):
            if random() <= p[i]:
                out.append(i)
        return out


class PBM:
    """Position Based Click Model
    ==========================
    
    Click model that simulates user interaction
    based on rank and document score.
    
    Two sets of parameters are learned, `alphas` and `gammas`.
    Parameters in `alphas` represent the attractiveness of documents
    given a certain query, while parameters in `gammas` represent
    the chance of viewing a document at a specific rank.
    User interaction is simulated by comparing a random value
    drawn from a uniform distribution against the product of
    one of the parameters in `gammas` and a value `epsilon`.
    Here `epsilon` represents the chance of clicking on a document
    even though the document is irrelevant and vice versa (`epsilon` replaces
    the parameters in `alpha` due to data sparcity).
    """
    def __init__(self):
        """Initializes class parameters.
        """
        self.alphas = {}
        self.gammas = []
    
    def update(self, item, alpha_sum, gamma_sum, prev_q, clicks, n=-1):
        """Updates sum of alpha and gamma contributions
        for the previous query database item.
        
        Parameters
        ----------
        item : dict
            A dictionary representing a database item.
        alpha_sum : dict
            A dictionary containing the summed contributions and
            number of contributions for all alphas corresponding
            to a document and query pair.
        gamma_sum : array_like
            A list containing the summed contributions of all gammas
            corresponding to a rank.
        prev_q : dict
            A dictionary representing the previous query database item.
        clicks : array_like
            A list of documents ids, returned by the last query,
            that the user clicked on.
        n : int
            Maximum rank for which parameters are learned.
        
        Returns
        -------
        alpha_sum : dict
            A dictionary containing the summed contributions and
            number of contributions for all parameters in `alphas`
            corresponding to a document and query pair.
        gamma_sum : array_like
            A list containing the summed contributions for all
            parameters in `gammas` corresponding to a rank.
        prev_q : dict
            A dictionary representing the previous query database item.
        clicks : array_like
            A list of documents ids, returned by the last query,
            that the user clicked on.
        """
        if item['a'] == 'q':
            q_id = prev_q['a_id']
            q_urls = prev_q['urls']
            if n >= 0:
                q_urls = q_urls[:n]
            gamma_length = len(q_urls)
            # Extend gammas and gamma_sum.
            while len(self.gammas) < gamma_length:
                self.gammas.append(random())
            while len(gamma_sum) < gamma_length:
                gamma_sum.append(0)
            for r, _id in enumerate(q_urls):
                uq = str((_id, q_id))
                # Extend alphas and alpha_sum.
                if self.alphas.get(uq) == None:
                    self.alphas[uq] = random()
                if alpha_sum.get(uq) == None:
                    alpha_sum[uq] = {'sum' : 0, 'length' : 0}
                # Update alphs_sum and gamma_sum.
                if _id in clicks:
                    alpha_sum[uq]['sum'] += 1
                    gamma_sum[r] += 1
                else:
                    alpha_sum[uq]['sum'] += \
                        (1 - self.gammas[r]) * self.alphas[uq] \
                        / (1 - self.gammas[r] * self.alphas[uq])
                    gamma_sum[r] += \
                        self.gammas[r] * (1 - self.alphas[uq]) \
                        / (1 - self.gammas[r] * self.alphas[uq])
                alpha_sum[uq]['length'] += 1
            prev_q = item
            clicks = []
        else:
            # Record clicked document id.
            clicks.append(item['a_id'])
        return alpha_sum, gamma_sum, prev_q, clicks
    
    def _learn(self, database, n=-1):
        """Learns class parameters for one run over the given database.
        
        Parameters
        ----------
        database : array_like
            Array of dictionaries representing database items.
        n : int
            Maximum rank for which parameters are learned.
        
        Returns
        -------
        None
        """
        alpha_sum = {}
        gamma_sum = []
        prev_q = database[0]
        session_id = prev_q['id']
        clicks = []
        
        empty_q = {'id' : -1, 'a' : 'q', 'a_id' : -1, 'urls' : []}
        
        # Adjust query counter for empty query.
        n_queries = 0
        if prev_q['a'] != 'q':
            n_queries -= 1
        
        # Sum alpha and gamma contributions
        # of each item in the database.
        for item in database[1:] + [empty_q]:
            if item['a'] == 'q':
                n_queries += 1
            if session_id != item['id']:
                alpha_sum, gamma_sum, prev_q, clicks = self.update(
                    empty_q, alpha_sum, gamma_sum, prev_q, clicks, n)
                session_id = item['id']
            alpha_sum, gamma_sum, prev_q, clicks = self.update(
                item, alpha_sum, gamma_sum, prev_q, clicks, n)
        
        # Update alphas and gammas.
        for uq, alpha in alpha_sum.items():
            self.alphas[uq] = (alpha['sum'] + 1) \
                / float(alpha['length'] + 2)
        for r, gamma_sum_r in enumerate(gamma_sum):
            self.gammas[r] = (gamma_sum_r + 1) / float(n_queries + 1)
        
        return
    
    def learn(self, database, n_decimals, n_consecutive, n_rank=-1):
        """Learns class parameters on the given database
        until convergence.
        
        Parameters
        ----------
        database : array_like
            Array of dictionaries representing database items.
        n_decimals : int
            Number of decimals on which convergence is checked.
        n_consecutive : int
            Number of consecutive database iterations for which
            convergence is checked.
        n_rank : int
            Maximum rank for which parameters are learned.
        
        Returns
        -------
        None
        """
        prev_gammas = []
        convergence = False
        while convergence == False:
            self._learn(database, n_rank)
            prev_gammas.append(
                [round(gamma, n_decimals) for gamma in self.gammas])
            if len(prev_gammas) >= n_consecutive:
                while len(prev_gammas) > n_consecutive:
                    prev_gammas.pop(0)
                convergence = True
                for prev, cur in zip(prev_gammas[:-1], prev_gammas[1:]):
                    for gamma_prev, gamma_cur in zip(prev, cur):
                        if gamma_prev - gamma_cur != 0:
                            convergence = False
                            break
                    if convergence == False:
                        break
        return
    
    def get_p(self, relevance_grades, epsilon=1e-1):
        """Determines chance of clicking on a document.
        
        Parameters
        ----------
        relevance_grades : array_like
            Array containing relevance grades for all documents
            returned by a search query.
        epsilon : float
            Value representing the chance of clicking on a document
            even though the document is irrelevant and vice versa.
        
        Returns
        -------
        out : list
            List of probabilities corresponding
            to entries in `search_results`.
        """
        out = []
        for i in range(min(len(self.gammas), len(relevance_grades))):
            if relevance_grades[i] == 0:
                out.append(self.gammas[i] * epsilon)
            else:
                out.append(self.gammas[i] * (1 - epsilon))
        return out
    
    def get_clicks(self, relevance_grades, epsilon=1e-1):
        """Simulate user interaction by determining
        what documents are clicked on.
        
        Parameters
        ----------
        relevance_grades : array_like
            Array containing relevance grades for all documents
            returned by a search query.
        epsilon : float
            Value representing the chance of clicking on a document
            even though the document is irrelevant and vice versa.
        
        Returns
        -------
        out : list
            List of indices of the documents that were clicked on
            in the simulation.
        """
        p = self.get_p(relevance_grades)
        out = []
        for i in range(len(p)):
            if random() <= p[i]:
                out.append(i)
        return out


def main():
    database = read_yandex('./YandexRelPredChallenge.txt')
    cm = PBM()
    cm.learn(database, n_decimals=3, n_consecutive=5, n_rank=3)
    print('Gamma\'s\n-------\n  ' + str(cm.gammas))
    # Simulate clicking.
    relevance_grades = [0,0,1]
    counter = [0 for _ in range(len(search_results))]
    for _ in range(1000):
        for i in cm.get_clicks(relevance_grades):
            counter[i] += 1
    plt.bar(range(len(counter)), counter)
    plt.show()


if __name__ == '__main__':
    main()


