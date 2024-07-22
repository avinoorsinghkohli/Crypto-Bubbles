from torch.utils.data import Dataset
import os
import pickle as pkl
import torch
import networkx as nx
from torch_geometric import utils

class GraphDataset(Dataset):
    def __init__(self, path_graph_data, path_coin_data, lookback=5, lookahead=10, max_tweets_per_day=7):
        self.path_graph_data = path_graph_data
        self.file_names = sorted(os.listdir(path_graph_data))
        self.lookback=lookback
        self.lookahead=lookahead
        self.MAX_TWEETS_PER_DAY = max_tweets_per_day

        with open(path_coin_data, 'rb') as f:
            self.all_coins, self.coin2idx, _, self.idx2date = pkl.load(f)
        
        with open('data/black_listed_dates.pkl', 'rb') as f:
            self.black_listed_dates = set(pkl.load(f))
        
        # self.coin2idx = {coin: idx for idx, coin in enumerate(list(self.coin2idx)[:306])}

    def __len__(self):
        return len(self.idx2date)
    
    def __getitem__(self, idx):
        date = self.idx2date[idx]
        # try:
        
        with open(self.path_graph_data + f"/{date}.pkl", "rb") as f:
            data_dict = pkl.load(f)


        embed = torch.zeros(self.lookback, len(self.coin2idx.keys()), self.MAX_TWEETS_PER_DAY, 768)
        graphs = []
        time_feats = torch.zeros(self.lookback, len(self.coin2idx.keys()), self.MAX_TWEETS_PER_DAY, 1)

        lookback_dates = data_dict["bubble_data"]["lookback_dates"]
        lookback_dates.sort()
        
        for idx_, past_date in enumerate(lookback_dates):
            with open(self.path_graph_data + f"/{date}.pkl", "rb") as f:
                past_data = pkl.load(f)
            embed[idx_] = past_data["embeddings"]
            time_feats[idx_] = past_data["time_feats"]
            graphs.append(past_data["graphs"])


        graphs = [utils.from_scipy_sparse_matrix(nx.to_scipy_sparse_matrix(G))[0] for G in graphs]
        
        start_idx = torch.zeros((len(self.all_coins), 10))
        end_idx = torch.zeros((len(self.all_coins), 10))
        num_bubbles = torch.zeros((len(self.all_coins), 1))
        true_bubble = torch.zeros((len(self.all_coins), 10))

        for coin, coin_data in data_dict["bubble_data"].items():
            if coin != 'lookback_dates':
                coin = self.coin2idx[coin]
                start_idx[coin] = torch.Tensor(coin_data["lookahead_starts"])
                end_idx[coin] = torch.Tensor(coin_data["lookahead_ends"])
                num_bubbles[coin][0] = coin_data["n_bubbles"]
                true_bubble[coin] = torch.Tensor(coin_data["bubble"])

        return (
            embed,
            start_idx,
            end_idx,
            num_bubbles,
            true_bubble,
            graphs,
            time_feats
        )
        # except Exception as e:
        #     print(e)
        
        
if __name__ == '__main__':
    dataset = GraphDataset('data/test_data_graphs_only_lookback_5_lookahead_10_stride_3/', 'data/coin_metadata_test.pkl', max_tweets_per_day=15)

    item = dataset.__getitem__(0)
    print(item)