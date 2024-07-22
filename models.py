from _models import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.nn import GATConv

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(LSTMEncoder, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, batch_first=True)

    def forward(self, src):
        outputs, (hidden, cell) = self.rnn(src)

        return hidden.squeeze(0), cell.squeeze(0)


class LSTMEncoderAttn(nn.Module):
    def __init__(self, input_dim, hid_dim, maxlen=75):
        super(LSTMEncoderAttn, self).__init__()
        self.rnn = nn.LSTM(input_dim, hid_dim, batch_first=True)
        self.attn = SimpleAttn(hid_dim, maxlen=maxlen, use_attention=True)

    def forward(self, src, len_feats):
        outputs, (hidden, cell) = self.rnn(src)
        hidden = hidden.permute(1, 0, 2)
        cell = cell.permute(1, 0, 2)

        hidden = self.attn(outputs, hidden, len_feats)
        cell = self.attn(outputs, cell, len_feats)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, input_dim, hid_dim, num_span_classes, out_dim=2, bs=16, num_days=20
    ):
        super(Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.num_days = num_days
        self.input_dim = input_dim
        self.rnn_cell = nn.LSTMCell(input_dim, hid_dim)
        self.fc_in = nn.Linear(hid_dim, input_dim)
        self.fc_out = nn.Linear(hid_dim, out_dim)
        self.num_span_layer = nn.Linear(hid_dim, num_span_classes)
        self.LogSoftmax = nn.LogSoftmax(dim=-1)
        self.Softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()

    def forward(self, hx, cx):
        """
        hx: (batch_size, hiddem_dim)
        cx: (batch_size, hiddem_dim)
        """
        bs, hid_dim = hx.shape
        num_spans = self.Softmax(self.num_span_layer(hx))

        input = torch.zeros(size=(bs, self.input_dim)).cuda()
        outputs = []
        for i in range(self.num_days):
            hx, cx = self.rnn_cell(input, (hx, cx))
            input = self.relu(self.fc_in(hx))
            # print(hx.shape)
            outputs.append(self.Softmax(self.fc_out(hx)))

        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.permute(1, 0, 2)
        # outputs.shape = (batch_size, num_days, 2)
        return num_spans, outputs


class TimeLSTMEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, bs):
        """
        return_last -> to return only the last hidden/cell state
        """
        super(TimeLSTMEncoder, self).__init__()
        self.rnn = TimeLSTM(input_dim, hid_dim)
        self.bs = bs
        self.hid_dim = hid_dim
        self.device = torch.device("cuda:0")

    def init_hidden(self, bs):
        h = Variable(torch.zeros(bs, self.hid_dim)).to(self.device)
        c = Variable(torch.zeros(bs, self.hid_dim)).to(self.device)

        return (h, c)

    def forward(self, sentence_feats, time_feats):
        """
        sentence_feat = (B*75*N)
        time_feats = (B*75)
        """

        h_init, c_init = self.init_hidden(sentence_feats.shape[0])
        lstmout, (h_out, c_out) = self.rnn(sentence_feats, time_feats, (h_init, c_init))
        return h_out, c_out


class TimeLSTMAttnEncoder(nn.Module):
    def __init__(self, input_dim, hid_dim, bs, maxlen=75):
        super(TimeLSTMAttnEncoder, self).__init__()
        self.rnn = TimeLSTM(input_dim, hid_dim)
        self.attn = SimpleAttn(hid_dim, maxlen=maxlen, use_attention=True)
        self.bs = bs
        self.hid_dim = hid_dim
        self.device = torch.device("cuda:0")

    def init_hidden(self, bs):
        h = Variable(torch.zeros(bs, self.hid_dim)).to(self.device)
        c = Variable(torch.zeros(bs, self.hid_dim)).to(self.device)

        return (h, c)

    def forward(self, sentence_feats, time_feats, len_feats = None):
        """
        sentence_feat = (B*75*N)
        time_feats = (B*75)
        len_feats = (B)
        """

        h_init, c_init = self.init_hidden(sentence_feats.shape[0])

        lstmout, (h_out, c_out) = self.rnn(sentence_feats, time_feats, (h_init, c_init))
        # print(lstmout.shape)
        # print(h_out.shape)
        # print(len_feats.shape)

        h_out = self.attn(lstmout, h_out.unsqueeze(1), len_feats)
        c_out = self.attn(lstmout, c_out.unsqueeze(1), len_feats)

        return h_out, c_out


class HAttnLSTM(nn.Module):
    def __init__(
        self,
        text_dim,
        hid_dim,
        gfeat_dim,
        num_span_classes,
        lookback,
        lookahead,
        bs,
        nheads,
        dropout,
        intra_maxlen=75,
        inter_maxlen=5,
    ) -> None:
        super(HAttnLSTM, self).__init__()
        self.input_dim = text_dim
        self.hid_dim = hid_dim
        self.gfeat_dim = gfeat_dim
        self.lookback = lookback

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.intra_rnn = TimeLSTMAttnEncoder(text_dim, hid_dim, bs, maxlen=intra_maxlen)
        self.intra_linear = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
        )

        self.gat = GATConv(hid_dim, gfeat_dim, heads=nheads, negative_slope=0.2,concat=False, dropout=dropout)
        self.gat_linear=nn.Sequential(
            nn.Linear(gfeat_dim, gfeat_dim),
        )
        # self.intra_rnn = LSTMEncoderAttn(text_dim, hid_dim, maxlen=15)
        self.inter_rnn = LSTMEncoderAttn(gfeat_dim, hid_dim, maxlen=None)

        self.linear1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
        )
        self.relu = nn.ReLU()

        self.decoder = Decoder(
            hid_dim, hid_dim, num_span_classes, out_dim=2, bs=bs, num_days=lookahead
        )

    def forward(self, text_feat, time_feat, graphs):
        """
        text_feat: (B*75*D)
        time_feat: (B*75)
        graphs: (B)
        """
        # text_feat = text_feat.view(-1, self.lookback, 15, self.input_dim)
        # time_feat = time_feat.view(-1, self.lookback, 15)
        text_feat = torch.squeeze(text_feat)
        time_feat = torch.squeeze(time_feat)
        inter_input = []
        # import pdb; pdb.set_trace()
        for i in range(self.lookback):
            temp, _ = self.intra_rnn(text_feat[i], time_feat[i])
            # temp, _ = self.intra_rnn(text_feat[:, i], len_feats[:, i])
            inter_input.append(torch.unsqueeze(self.intra_linear(temp), 0))
            # print(torch.sum(torch.isnan(temp)))

        gat_input = torch.cat(inter_input)
        # print(f'gat_input shape = {gat_input.shape}')
        inter_input = []
        for i, graph in enumerate(graphs):
            inter_input.append(torch.unsqueeze(self.gat(gat_input[i], graph[0]), 0))
    	
        inter_input = self.gat_linear(
            torch.cat(inter_input).view(-1, self.lookback, self.gfeat_dim)
        )
        # print(f'inter_input shape = {inter_input.shape}')
        hout, cout = self.inter_rnn(inter_input, None)
        # print(F'hout shape = {hout.shape}')
        # print(F'cout shape = {cout.shape}')

        hout = self.relu(self.linear1(hout))
        cout = self.relu(self.linear2(cout))

        num_spans, outputs = self.decoder(hout, cout)
        return num_spans, outputs
