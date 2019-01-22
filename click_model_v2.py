#!/usr/bin/env python3

from random import random
import matplotlib.pyplot as plt


def read_yandex(path, n=-1):
    out = []
    with open(path) as f:
        for i, l in enumerate(f):
            if n > 0 and i > n:
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
    def __init__(self):
        self.rho = 1.0
    
    def learn(self, database):
        n_clicks, n_docs = 0, 0
        for item in database:
            if item['a'] == 'q':
                n_docs += len(item['urls'])
            else:
                n_clicks += 1
        self.rho = n_clicks / float(n_docs)
        return
    
    def get_p(self, search_results):
        return [self.rho for _ in range(len(search_results))]
    
    def get_clicks(self, search_results):
        p = self.get_p(search_results)
        clicked = []
        for i, doc in enumerate(search_results):
            if random() <= p[i]:
                clicked.append(i)
        return clicked


class PBM:
    """DOES NOT LEARN ALPHA"""
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.gammas = []
    
    def update_item(self, item, gamma_sum, prev_q, clicks):
        if item['a'] == 'q':
            while len(self.gammas) < len(prev_q['urls']):
                self.gammas.append(random())
                gamma_sum.append(0)
            for r, _id in enumerate(prev_q['urls']):
                if _id in clicks:
                    gamma_sum[r] += 1
                else:
                    gamma_sum[r] += self.gammas[r] * self.epsilon \
                        / (1 - self.gammas[r] * (1 - self.epsilon))
            prev_q = item
            clicks = []
        else:
            clicks.append(item['a_id'])
        return gamma_sum, prev_q, clicks
    
    def update_session(self, session, gamma_sum, n_queries):
        if len(session) < 1:
            return gamma_sum, n_queries
        session_length = 1
        gamma_sum_session = [0 for _ in range(len(self.gammas))]
        prev_q = session[0]
        clicks = []
        for item in session[1:]:
            if item['a'] == 'q':
                session_length += 1
            gamma_sum_session, prev_q, clicks = self.update_item(
                item, gamma_sum_session, prev_q, clicks)
        gamma_sum_session, _, _ = self.update_item(
            {'a' : 'q'}, gamma_sum_session, prev_q, clicks)
        n_queries += session_length
        for r, gamma_sum_r in enumerate(gamma_sum_session):
            if len(gamma_sum) < r + 1:
                gamma_sum.append(0)
#            gamma_sum[r] += gamma_sum_r
            gamma_sum[r] += gamma_sum_r / float(session_length)
#            self.gammas[r] = gamma_sum_r / float(session_length)
#        print('>>',self.gammas)
        return gamma_sum, n_queries
    
    def learn(self, database):
        session_id = -1
        session = []
        gamma_sum = []
        n_sessions = 0
        n_queries = 0
        for item in database:
            if session_id != item['id']:
                gamma_sum, n_queries = self.update_session(session, gamma_sum, n_queries)
                session_id = item['id']
                session = []
                n_sessions += 1
            session.append(item)
        gamma_sum, n_queries = self.update_session(session, gamma_sum, n_queries)
        
        for r, gamma_sum_r in enumerate(gamma_sum):
#            self.gammas.append(x / float(n_queries))
            self.gammas[r] = gamma_sum_r / float(n_sessions)
        
        print(self.gammas)
        
        return
    
    def get_p(self, search_results):
        return []
    
    def get_clicks(self, search_results):
        return []


def main():
    database = read_yandex('./YandexRelPredChallenge.txt')
    cm = PBM()
    for _ in range(10):
        cm.learn(database)
    search_results = [0 for _ in range(20)]
    counter = [0 for _ in range(len(search_results))]
    for _ in range(1):
        for i in cm.get_clicks(search_results):
            counter[i] += 1
#    plt.bar(range(len(counter)), counter)
#    plt.show()

if __name__ == '__main__':
    main()


