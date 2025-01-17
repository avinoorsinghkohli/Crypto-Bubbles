import os
import torch
import torch.nn as nn
from torch import optim
import argparse
import time
import logging
import copy
from utils import summarize_results, callback_get_label
# from focal_loss.focal_loss import FocalLoss
# from torchsampler import ImbalancedDatasetSampler
from dataset import GraphDataset
from torch.utils.data import DataLoader
from models import *

torch.multiprocessing.set_sharing_strategy("file_system")
start_time = str(time.strftime("%Y%m%d-%H%M%S"))
print(start_time)

if not os.path.exists("logs"):
    os.mkdir("logs")

if not os.path.exists("saved_models"):
    os.mkdir("saved_models")

logging.basicConfig(
    filename=f"logs/{start_time}.log", level=logging.INFO, format="%(message)s"
)

device = torch.device("cuda:0")
parser = argparse.ArgumentParser(
    description="Neural Graph Cryto Bubble Predictor -- Trainer"
)

parser.add_argument(
    "--data",
    default="text",
    type=str,
    help="Data to use for training [price, text] (default: simple)",
)
parser.add_argument(
    "--lr",
    default=0.003,
    type=float,
    help="Learning rate to use for training (default: 0.001)",
)
parser.add_argument(
    "--num_epochs",
    default=100,
    type=int,
    help="Number of epochs to run for training (default: 50)",
)

parser.add_argument(
    "--decay",
    default=1e-5,
    type=float,
    help="Weight decay to use for training (default: 1e-5)",
)
parser.add_argument(
    "--batch_size",
    default=1,
    type=int,
    help="Batch Size use for training the model (default: 16)",
)

parser.add_argument(
    "--num_lookahead",
    default=10,
    type=int,
    help="Number of Lookahead days (default: 10)",
)
parser.add_argument(
    "--data_lookahead",
    default=10,
    type=int,
    help="For loading the dataset (default: 10)",
)

parser.add_argument(
    "--num_lookback", default=5, type=int, help="Number of Lookback days (default: 5)",
)

parser.add_argument(
    "--hidden_dim",
    default=128,
    type=int,
    help="Number of Hidden Dims for LSTM (default: 8)",
)

parser.add_argument(
    "--do_sampling",
    default=False,
    action="store_true",
    help="Whether to do sampling or not (default: False)",
)
parser.add_argument(
    "--focal_loss",
    default=False,
    action="store_true",
    help="Whether to use any custom loss (default: False)",
)
parser.add_argument(
    "--stride", default="3", type=str, help="Stride of this file (default: 3)",
)


args = parser.parse_args()
print(args)

if args.data == "price":
    load_embeds = False
    input_dim = 1
elif args.data == "text":
    input_dim = 768
elif args.data == "text_time":
    input_dim = 768 + 1
else:
    raise NotImplementedError

batch_size = args.batch_size
num_epochs = args.num_epochs
num_days = args.num_lookahead
num_lookback = args.num_lookback
hidden_dim = args.hidden_dim
LR = args.lr
num_span_classes = 5

logging.info(
    f"Batch Size: {batch_size} Num Epochs: {num_epochs}  Hidden Dim: {hidden_dim} LR: {LR} Decay: {args.decay}\n "
)

logging.info(
    f"Data: {args.data}  Data used: [split]_data_price_only_lookback_{args.num_lookback}_lookahead_{args.data_lookahead}_stride_{args.stride}.pkl \n Num Lookback: {num_lookback} Num Lookahead: {num_days} Sampling: {args.do_sampling} Focal Loss: {args.focal_loss} \n\n"
)
# data_used = {
#     "train": [
#         f"train_data_price_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl",
#         f"train_data_text_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl",
#     ],
#     "val": [
#         f"test_data_price_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl",
#         f"test_data_text_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}.pkl",
#     ],
# }

data_used = {
    "train": f"data/train_data_graphs_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}",
    "val": f"data/test_data_graphs_only_lookback_{num_lookback}_lookahead_{args.num_lookahead}_stride_{args.stride}",
}
trainset = GraphDataset(
    path_graph_data=data_used["train"],
    path_coin_data="data/coin_metadata_train.pkl",
)

valset = GraphDataset(
    path_graph_data=data_used["val"],
    path_coin_data="data/coin_metadata_test.pkl",
    max_tweets_per_day=15
)

if args.do_sampling:
    trainloader = DataLoader(
        trainset,
        # sampler=ImbalancedDatasetSampler(
        #     trainset, callback_get_label=callback_get_label
        # ),
        batch_size=batch_size,
        num_workers=2,
        drop_last=True,
    )
else:
    trainloader = DataLoader(
        trainset, shuffle=True, batch_size=batch_size, num_workers=2, drop_last=True,
    )
valloader = DataLoader(
    valset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=True
)
dataloaders = {"train": trainloader, "val": valloader}

if args.data == "price":
    maxlen = num_lookback
else:
    maxlen = num_lookback * 15

model = HAttnLSTM(
    input_dim,
    hidden_dim,
    gfeat_dim=256,
    num_span_classes=num_span_classes,
    lookback=num_lookback,
    lookahead=10,
    bs=306,
    intra_maxlen=15,
    inter_maxlen=num_lookback,
    nheads=1,
    dropout=0.3
)
model = model.to(device)

dataset_sizes = {"train": len(trainset), "val": len(valset)}

if args.focal_loss:
    criterion1 = FocalLoss(alpha=2, gamma=5)
else:
    criterion1 = nn.BCELoss()

criterion2 = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=args.decay)


def train_model(criterion, ce_criterion, num_epochs=25):
    since = time.time()

    best_model_wts = [copy.deepcopy(model.state_dict())]
    best_mcc = -1
    results_dict = {"train": [], "val": []}

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        logging.info("Epoch {}/{}".format(epoch, num_epochs - 1))
        logging.info("-" * 10)

        start_idx_true_list = []
        start_idx_pred_list = []
        end_idx_true_list = []
        end_idx_pred_list = []
        true_bubble_list = []
        num_bubble_true_list = []
        num_bubble_pred_list = []
        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            for batch_data in dataloaders[phase]:
                # if args.data == "price":
                # inputs = batch_data[0].unsqueeze(
                # -1).to(torch.float).to(device)
                # elif args.data == "text":
                # else:
                # raise NotImplementedError

                if batch_data[0] == "Invalid":
                    print("Here")
                    continue

                inputs = batch_data[0].to(device).float()
                start_idx = batch_data[1].to(device).float()
                end_idx = batch_data[2].to(device).float()
                num_bubbles = batch_data[3].to(device).long()
                true_bubble = batch_data[4].to(device)
                graphs = batch_data[5]
                for i in range(len(graphs)):
                    graphs[i] = graphs[i].to(device)
                time_feats = batch_data[6].to(device).float()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    num_bubbles_preds, outputs = model(
                        inputs, time_feats, graphs
                    )
                    start_preds = outputs[:, :, 0]
                    end_preds = outputs[:, :, 1]

                    loss1 = criterion(start_preds, start_idx[0])
                    loss2 = criterion(end_preds, end_idx[0])
                    # print(num_bubbles_preds.shape, num_bubbles.shape)
                    loss3 = ce_criterion(num_bubbles_preds, torch.squeeze(num_bubbles))

                    loss = loss1 + loss2 + loss3

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                print(f"Epoch - {epoch} Batch Loss: {loss}")

                if phase == "val":
                    true_bubble_list.append(true_bubble)
                    start_idx_pred_list.append(start_preds)
                    end_idx_pred_list.append(end_preds)
                    num_bubble_true_list.append(num_bubbles)
                    num_bubble_pred_list.append(num_bubbles_preds)

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == "val":
                results = summarize_results(
                    true_bubble_list,
                    start_idx_pred_list,
                    end_idx_pred_list,
                    num_bubble_true_list,
                    num_bubble_pred_list,
                )
                results_dict[phase].append(results)

                logging.info(
                    "{} Loss: {:.4f} \t MCC: {:.4f} \t EM: {:.4f} EM (only_bubble): {:.4f} \n Accu (Span): {:.4f} \t Precision (Span): {:.4f} \t Recall (Span): {:.4f} \t F1 (Span): {:.4f} \n Acc (Bubble): {:.4f} \t Precision (Bubble): {:.4f} \t Recall (Bubble): {:.4f} \t F1 (Bubble): {:.4f}".format(
                        phase,
                        epoch_loss,
                        results["MCC"],
                        results["EM"],
                        results["EM_only_bubble"],
                        results["acc_span"],
                        results["precision_span"],
                        results["recall_span"],
                        results["f1_span"],
                        results["acc_bubble"],
                        results["precision_nbubble"],
                        results["recall_nbubble"],
                        results["f1_nbubble"],
                    )
                )
                mcc = results["MCC"]

                # deep copy the model
                if mcc > best_mcc:
                    best_mcc = mcc
                    best_f1_span = results["f1_span"]
                    best_accu_span = results["acc_span"]
                    best_em_only_bubble = results["EM_only_bubble"]
                    best_accu_num_bubbles = results["acc_bubble"]
                    best_f1_num_bubbles = results["f1_nbubble"]
                    best_model_wts = [
                        copy.deepcopy(model.state_dict()),
                    ]
                    torch.save(
                        {
                            "best_model_wts": best_model_wts[0],
                            #  "best_dec_wts": best_model_wts[1],
                            "best_val_mcc": best_mcc,
                            "args": args,
                            "model_name": "HAttnLSTM",
                            "dataused_path": data_used,
                            "results": results_dict,
                        },
                        "saved_models/"
                        + start_time
                        + f"_lookback{num_lookback}_lookahead_{num_days}.pkl",
                    )

    time_elapsed = time.time() - since
    logging.info(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    logging.info("Best test MCC: {:4f}".format(best_mcc))
    logging.info("Best test best_f1_span: {:4f}".format(best_f1_span))
    logging.info("Best test best_accu_span: {:4f}".format(best_accu_span))
    logging.info("Best test best_em_only_bubble: {:4f}".format(best_em_only_bubble))
    logging.info("Best test best_accu_num_bubbles: {:4f}".format(best_accu_num_bubbles))
    logging.info("Best test best_f1_num_bubbles: {:4f}".format(best_f1_num_bubbles))

    print("Best test MCC: {:4f}".format(best_mcc))
    print("Best test best_f1_span: {:4f}".format(best_f1_span))
    print("Best test best_accu_span: {:4f}".format(best_accu_span))
    print("Best test best_em_only_bubble: {:4f}".format(best_em_only_bubble))
    print("Best test best_accu_num_bubbles: {:4f}".format(best_accu_num_bubbles))
    print("Best test best_f1_num_bubbles: {:4f}".format(best_f1_num_bubbles))

    print(start_time)
    torch.save(
        {
            "best_model_wts": best_model_wts[0],
            "best_val_mcc": best_mcc,
            "args": args,
            "model_name": "HAttnLSTM",
            "dataused_path": data_used,
            "results": results_dict,
        },
        "saved_models/" + start_time + "_final.pkl",
    )
    return model


train_model(criterion1, criterion2, num_epochs)
