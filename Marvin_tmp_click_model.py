#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"Imports"
import numpy as np
import pandas as pd
import csv
import random

class PBM:
    def __init__(self, seed=42, epsilon=0.1):
        random.seed(seed)
        self.epsilon = epsilon
        self.gammas = []  #### Test values
        self.click_probabilities = []
    
        
    def position_based_click_model(self, interleaved_list):
        """Run Position Based Click Model.
        
        Parameters
        ----------
        interleaved_list : list of tuples
            Index of list elements represents the rank of that element. Each 
            element is a tuple of the relevance score and a binary value 
            representing whether the document comes from algorithm E (1) or P (0).
            
        Returns
        -------
        clicked_id : int
            Index of the ranking algorithm that is clicked.
        """
        
        if len(self.gammas) == 0:
            raise Exception('No gamma values, train the model first')
            
        self.pbm_probabilities(interleaved_list)
        
        return self.pbm_clicks(interleaved_list)
    
    def pbm_probabilities(self, interleaved_list):
        """Helper method: Predicts the click probability given a ranked list of 
        relevance labels
        
        Parameters
        ----------
        interleaved_list : list of tuples
            Index of list elements represents the rank of that element. Each 
            element is a tuple of the relevance score and a binary value 
            representing whether the document comes from algorithm E (1) or P (0).
            
        Returns
        -------
        click_probabilities   : List of floats 
            Click probabilities for each document in interleaved_list.
    
        """
        if len(self.gammas) == 0:
            raise Exception('No gamma values, train the model first')
            
        probabilities = []
        for i in range(len(interleaved_list)):
            if interleaved_list[i][1] == 1:
                attractiveness = 1-self.epsilon
            else: attractiveness = self.epsilon
            
            probability = self.gammas[i] *  attractiveness
            probabilities.append(probability)
            
        self.click_probabilities = probabilities
        
        return 
    
    def pbm_clicks(self, interleaved_list):
        """Helper method:  Decides stochastically whether a document is clicked 
        based on given probabilities
        
        Parameters
        ----------
        interleaved_list  : ....
            
        Returns
        -------
        clicked_id : int
            Index of the ranking algorithm that is clicked.
        
        """
        clicks = []
        while len(clicks) == 0:
            for i in range(len(interleaved_list)):
                if random.random() < self.click_probabilities[i]:
                    clicks.append(i)
                    
        count = 0
        for j in clicks:
            count += interleaved_list[j][1]
        
        if count/len(clicks) > 0.5:
            return 1
        else: 
            return 0
        
    
    def train_pbm(self, file="YandexRelPredChallenge.txt"):
        """Learns PBM gamma parameters based on training click data using
        Expectation Maximization
        
        Parameters
        ----------
            
        Returns
        -------
        gammas    : List of floats 
            Learned examination parameters. 
    
    
        """

        query_documents = {}
        clicked_doc_queries = {}
        order_queries = []

        with open(file) as csvfile:
            yandex = csv.reader(csvfile, delimiter='\t')
            for row in yandex:
                if row[2] == 'Q':

                    query_region_id = row[3] + row[4] # Latest query
                    doc_urls = query_documents.get(query_region_id)
                    if doc_urls == None:
                        doc_urls = set()

                    for log_url in row[5:15]:
                        doc_urls.add(log_url)

                    query_documents[query_region_id] = urls
                else:
                    clicked_doc = row[3]
                    

        """
        click_log = []
        with open(file) as csvfile:
            yandex = csv.reader(csvfile, delimiter='\t')
            for row in yandex:
                log = [row[0], 1 if row[2] == 'C' else 0, row[3]]
                if log[1] == 0:
                    log2 = [row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14]]
                    log.extend(log2)
                click_log.append(log)
        
        session = [l[0] for l in click_log]
        type_action = [l[1] for l in click_log]
        q_u_id = [l[2] for l in click_log]
        df = pd.DataFrame({'session': session, 'type_action': type_action, 'q_u_id': q_u_id}, dtype=int)
    
        queries = df[df.type_action == 0]        
        clicks = df[df.type_action == 1]

        alpha = 0
        gammas = [random.random() for i in range(6)]
        
        for i in range(10):
            for j in range(len(gammas)):
                count = 0
                counter = 0
                for k in range(len(df)):
                    if df.iloc[k].type_action == 0:
                        u_id = int(click_log[k][j + 3])
                        c_u = 0
                        if df[(df.session == df.iloc[k].session) & (df.type_action == 1) & (df.q_u_id == u_id)].empty:
                            c_u = 0
                        else:
                            c_u = 1
                            
                        count += c_u + ((1 - c_u) * (gammas[j] * (1 - self.epsilon)) / (1 - gammas[j] * self.epsilon))
                        counter += 1
                    
                gammas[j] = count / counter
            
            print("Training " + str((i+1)/(10) * 100)+ "% complete")
            print(gammas)
        
        self.gammas = gammas
        """
        return df, queries, clicks, click_log

class RCM:
    def __init__(self, seed=42):
        random.seed(seed)
        self.rho = 0
        
    def train_rho(self, file="YandexRelPredChallenge.txt"):
        click_log = []
        with open(file) as csvfile:
            yandex = csv.reader(csvfile, delimiter='\t')
            for row in yandex:
                log = [row[0], 1 if row[2] == 'C' else 0, row[3]]
                click_log.append(log)
        
        session = [l[0] for l in click_log]
        type_action = [l[1] for l in click_log]
        q_u_id = [l[2] for l in click_log]
        df = pd.DataFrame({'session': session, 'type_action': type_action, 'q_u_id': q_u_id}, dtype=int)
        queries = df[df.type_action == 0]        
        clicks = df[df.type_action == 1]
        
        self.rho = len(clicks)/(len(queries)*10)
        
        return
    
    def random_click_model(self, interleaved_list):
        # Probability of E winning not 0.5!!
        """Random click model.
        
        Parameters
        ----------
        interleaved_list : list of tuples
            Index of list elements represents the rank of that element. Each 
            element is a tuple of the relevance score and a binary value 
            representing whether the document comes from algorithm E (1) or P (0).
            
        Returns
        -------
        clicked_id : int
            Index of the ranking algorithm that is clicked.
        """
        if self.rho == 0:
            raise Exception('No rho value, train the model first')
            
        clicks = []
        while len(clicks) == 0:
            for i in range(len(interleaved_list)):
                if random.random() < self.rho:
                    clicks.append(i)
        
        count = 0
        for j in clicks:
            count += interleaved_list[j][1]
        
        if count/len(clicks) > 0.5:
            return 1
        else: 
            return 0
        
import time

rcm = RCM()
counter = 0
test = [(0,1),(0,0),(0,1),(0,0)]
rcm.train_rho()
for k in range(10000):
    counter += rcm.random_click_model(test)
print(counter/10000)

start = time.time()
            
pbm = PBM()
df, queries, clicks, click_log = pbm.train_pbm()

end = time.time()
print(end - start)
    