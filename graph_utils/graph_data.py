from datetime import datetime, timedelta
from time import time
import pandas as pd
import numpy as np
import os
import pickle as pkl
import json
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
import torch

ROOT_DIR = '../data/tweets/'
WINDOW_SIZE = 5

def get_coins_list(filename=None):
    if filename is not None:
        with open(filename, 'r') as file:
            coins_loaded = json.load(file)
        return coins_loaded
    
    coins = list(filter(lambda x: len(x) > 0, [subdir.split('/')[-1] for subdir, dirs, files in os.walk(ROOT_DIR)]))
    with open('coins_list.json', 'w') as file:
            json.dump(coins, file)
    return coins

coins = get_coins_list(filename='coins_list.json')
coin_map = {coin: idx for idx, coin in enumerate(coins)}

def get_date_string(d):
    return d.strftime('%Y-%m-%d')

def get_adjacency_matrix(date):
    mat = np.zeros((len(coins), len(coins)))
    for idx_u, coin_u in enumerate(coins):
        filepath_u = os.path.join(ROOT_DIR, coin_u, get_date_string(date) + '.csv')
        try:
            df = pd.read_csv(filepath_u, engine='python')
        except:
            # TODO Handle if data for given date not found or file is empty
            continue
            pass
        if df.shape[0] == 0:
            # TODO Handle if no data in file
            continue
        tweet_key = 'tweet'
        if tweet_key not in df.columns:
            tweet_key = 'Text'

        total = 0
        for tweet in df[tweet_key]:
            if tweet is None:
                continue
            total += 1
            tweet = tweet.upper()
            regex = r"\$.\w*?\b"
            tickers = re.findall(regex, tweet)
            for ticker in np.unique(tickers):
                if ticker[1:] in coin_map:
                    mat[coin_map[coin_u]][coin_map[ticker[1:]]] += 1
        mat[coin_map[coin_u], :] /= total
    
    np.save('temp.npy', mat)
    return mat
            

def make_matrix(start_date, last_date, window_size=WINDOW_SIZE):
    ROOT_DIR = '../tweets/'
    mat = np.zeros((len(coins), len(coins)))

    start_date = datetime(start_date[0], start_date[1], start_date[2])
    end_date = start_date + timedelta(days=window_size - 1)
    last_date = datetime(last_date[0], last_date[1], last_date[2])

    cur_date = start_date
    window_mats = []
    while cur_date <= end_date:
        mat = get_adjacency_matrix(cur_date)
        window_mats.append(mat)
        cur_date += timedelta(days=1)
    
    pbar = tqdm(total=(last_date - end_date).days)
    final_matrix = [np.array(window_mats)]
    while end_date < last_date:
        window_mats.pop(0)
        end_date += timedelta(days=1)
        start_date += timedelta(days=1)
        mat = get_adjacency_matrix(end_date)
        window_mats.append(mat)
        final_matrix.append(np.array(window_mats))
        pbar.update(1)
    final_matrix = np.array(final_matrix)
    print(final_matrix.shape)
    np.save('coin_coin_matrix_windowed', final_matrix)

def get_node_embeddings(date, model):
    embedding_map = {}
    for coin in coins:
        filepath = os.path.join(ROOT_DIR, coin, get_date_string(date) + '.csv')
        try:
            df = pd.read_csv(filepath, engine='python')
        except:
            # TODO Handle if data for given date not found or file is empty
            embedding_map[coin] = np.zeros(384)
            continue

        tweet_key = 'tweet'
        if tweet_key not in df.columns:
            tweet_key = 'Text'

        total = 0
        global_embedding = np.zeros(384)
        tweets = df[tweet_key].to_numpy()
        
        if len(tweets) > 15:
            random_indices = np.random.permutation(len(tweets))[:15]
            tweets = tweets[random_indices]
        
        for tweet in tweets:
            if tweet is None:
                continue
            total += 1
            embedding = model.encode(tweet)
            if np.isnan(embedding).astype(np.int32).sum() > 0:
                print(coin)
            global_embedding = global_embedding + embedding
        
        if total != 0:
            global_embedding = global_embedding / total
        embedding_map[coin] = global_embedding
    
    return embedding_map

def make_node_embeddings(start_date, last_date):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    start_date = datetime(start_date[0], start_date[1], start_date[2])
    last_date = datetime(last_date[0], last_date[1], last_date[2])

    cur_date = start_date
    window_mats = []
    pbar = tqdm(total=(last_date - start_date).days)
    embedding_date_map = {}
    while cur_date <= last_date:
        embedding_map = get_node_embeddings(cur_date, model)
        embedding_date_map[cur_date.strftime('%Y-%m-%d')] = embedding_map
        cur_date += timedelta(days=1)
        pbar.update(1)
    

def make_data(start_date, last_date):
    start_date = datetime(start_date[0], start_date[1], start_date[2])
    last_date = datetime(last_date[0], last_date[1], last_date[2])

    pbar = tqdm(total=(last_date - start_date).days)
    embedding_date_map = {}
    while start_date <= last_date:
        embedding_map = get_node_embeddings(cur_date, model)

    



make_node_embeddings([2019, 1, 1], [2019, 4, 12])