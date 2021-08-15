"""
Most of the codes are borrowed from
https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py
"""
import copy
import json
import math
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from configs import EMOTION_CATES



############## activation functions ##############
##################################################
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {
    'relu': nn.ReLU,
    'swish': swish,
    'gelu': gelu
}


############## the main model ####################
##################################################
class LMModel(nn.Module):
    """ Transformer with language model head only """
    def __init__(self, cfg, n_vocab, n_special, n_ctx):
        super(LMModel, self).__init__()
        n_indexs = n_vocab + n_special + n_ctx
        self.transformer = TransformerModel(cfg, n_indexs, n_ctx)
        self.lm_head = LMHead(self.transformer, cfg, n_vocab)

    def forward(self, x, pre_sts=None, last_only=False):
        """
        :param x: the input sequence
        :param pre_sts: to prevent recalculation. the prefix sequence's hidden states and Q,V saved in previous turn.
        :param last_only: only calculate the logits for the last position (only for generation)
        :return: logits of vocab words
        """
        sts = self.transformer(x, pre_sts)
        lm_logits = self.lm_head(sts[-1][0], last_only)
        return lm_logits, sts


class ELMModel(LMModel):
    """ Transformer with language model head and Emphathy head """
    def __init__(self, cfg, n_vocab, n_special, n_ctx, indexer, beta=1.0, init_std=0.02, tieSL=False):
        super(ELMModel, self).__init__(cfg, n_vocab, n_special, n_ctx)
        self.indexer = indexer
        self.beta = beta
        # emotion embeddings for Speaker and Listener
        n_emo = len(EMOTION_CATES)
        self.ES = nn.Linear(n_emo, cfg.n_emo_embd, bias=False)
        self.EL = nn.Linear(n_emo, cfg.n_emo_embd, bias=False)
        # word emotion embeddings for Speaker and Listener
        self.VS = nn.Linear(cfg.n_emo_embd, n_vocab+2, bias=False)  # and SOS, EOS
        self.VL = nn.Linear(cfg.n_emo_embd, n_vocab+2, bias=False)
        # dropout
        self.drop = nn.Dropout(cfg.embd_pdrop)
        # weight init
        nn.init.normal_(self.ES.weight, std=init_std)
        nn.init.normal_(self.EL.weight, std=init_std)
        nn.init.normal_(self.VS.weight, std=init_std)
        nn.init.normal_(self.VL.weight, std=init_std)
        if tieSL:
            self.EL.weight = self.ES.weight
            self.VL.weight = self.VS.weight
        # emotion classifier
        self.clf_head = ClfHead(indexer.EOS_IDX, cfg, n_emo)

    def forward(self, clf_idx, x, pre_sts=None, last_only=False):
        sts = self.transformer(x, pre_sts)
        last_hs = sts[-1][0]  # hidden states of the last layer
        # lm
        lm_logits = self.lm_head(last_hs, last_only)
        device = next(self.parameters()).device
        if type(clf_idx) == list:
            # here clf_idx is the specified emotions
            p_emo = torch.zeros((x.shape[0], len(EMOTION_CATES)))
            p_emo[torch.arange(x.shape[0]), clf_idx] = 1.0
            p_emo = p_emo.to(device)
            clf_logits = None
        else:
            # predict emotion prob distribution
            clf_sts = last_hs[torch.arange(x.shape[0]), clf_idx.to(device)]
            clf_logits = self.clf_head(clf_sts)
            p_emo = F.softmax(clf_logits, dim=-1)
        # calculate bias
        em_bias = torch.zeros(lm_logits.shape).to(device)
        x_ds = x[:, -1:, 2] if last_only else x[:, :, 2]  # dialog states
        for i in range(x.shape[0]):
            pe = p_emo[i:i+1]
            # the averaged emotion vector
            es = self.ES(pe)
            el = self.EL(pe)
            bias_s = self.VS(self.drop(es))
            bias_l = self.VL(self.drop(el))
            em_bias[i, x_ds[i]==self.indexer.DS_SPEAKER_IDX] = bias_s
            em_bias[i, x_ds[i]==self.indexer.DS_LISTENER_IDX] = bias_l
        lm_logits = lm_logits + self.beta * em_bias
        return lm_logits, sts, clf_logits


############## components ########################
##################################################
class LMHead(nn.Module):
    """ Language Model Head for the transformer """
    def __init__(self, model, cfg, n_vocab):
        super(LMHead, self).__init__()
        self.n_embd = cfg.n_embd
        embed_shape = model.embed.weight.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model.embed.weight  # Tied weights
        self.n_decoding_vocab = n_vocab + 2  # and SOS, EOS

    def forward(self, h, last_only=False):
        if last_only:
            h = h[:, -1:, :]
        lm_logits = self.decoder(h)
        return lm_logits[:, :, :self.n_decoding_vocab]


class ClfHead(nn.Module):
    def __init__(self, clf_token, cfg, n_class):
        super(ClfHead, self).__init__()
        self.clf_token = clf_token
        self.dropout = nn.Dropout(cfg.clf_pdrop)
        n_hs = [cfg.n_embd] + cfg.clf_hs + [n_class]
        mlp = []
        for i in range(len(n_hs)-1):
            n_input = n_hs[i]
            n_output = n_hs[i+1]
            l = nn.Linear(n_input, n_output)
            nn.init.normal_(l.weight, std=0.02)
            nn.init.normal_(l.bias, 0)
            mlp.append(l)
        self.mlp = nn.ModuleList(mlp)

    def forward(self, h):
        for i in range(len(self.mlp)-1):
            layer = self.mlp[i]
            h = self.dropout(h)
            h = layer(h)
            h = F.relu(h)
        output_layer = self.mlp[-1]
        clf_logits = output_layer(self.dropout(h))
        return clf_logits


class TransformerModel(nn.Module):
    """ Transformer model """
    def __init__(self, cfg, n_indexs, n_ctx):
        super(TransformerModel, self).__init__()
        self.embed = nn.Embedding(n_indexs, cfg.n_embd)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        block = Block(n_ctx, cfg, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(cfg.n_layer)])
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, x, pre_sts=None):
        # for training mode
        if pre_sts is None:
            e = self.drop(self.embed(x))
            # Add the position information to the input embeddings
            h = e.sum(dim=2)
        # for eval mode
        else:
            # get newly added words' embeddings
            prev_len = pre_sts[0].size(1)
            e_new = self.drop(self.embed(x[:, prev_len:, :]))
            h_new = e_new.sum(dim=2)
            h = torch.cat([pre_sts[0], h_new], dim=1)
        # record the output hidden states of each layer
        sts = []
        sts.append(h)
        for i, block in enumerate(self.h):
            # pre_h the prefix sequence's hidden states of the current layer
            pre_h, pre_k, pre_v = pre_sts[i+1] if pre_sts is not None else (None, None, None)
            h, k, v = block(h, pre_h, pre_k, pre_v)
            sts.append((h, k, v))
        return sts


class Block(nn.Module):
    def __init__(self, n_ctx, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg.n_embd
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = LayerNorm(nx)
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = LayerNorm(nx)

    def forward(self, x, pre_h=None, pre_key=None, pre_value=None):
        l_needed = x.shape[1] if pre_h is None else x.shape[1] - pre_h.shape[1]
        x_needed = x[:, -l_needed:, :]
        a, key, value = self.attn(x_needed, pre_key, pre_value)
        n = self.ln_1(x_needed + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        if pre_h is not None:
            h = torch.cat([pre_h, h], dim=1)
        return h, key, value


class LayerNorm(nn.Module):
    "Construct a layernorm module in the OpenAI style (epsilon inside the square root)."

    def __init__(self, n_state, e=1e-5):
        super(LayerNorm, self).__init__()
        self.g = nn.Parameter(torch.ones(n_state))
        self.b = nn.Parameter(torch.zeros(n_state))
        self.e = e

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.e)
        return self.g * x + self.b


class Conv1D(nn.Module):
    def __init__(self, nf, rf, nx):
        super(Conv1D, self).__init__()
        self.rf = rf
        self.nf = nf
        if rf == 1:  # faster 1x1 conv
            w = torch.empty(nx, nf)
            nn.init.normal_(w, std=0.02)
            self.w = Parameter(w)
            self.b = Parameter(torch.zeros(nf))
        else:  # was used to train LM
            raise NotImplementedError

    def forward(self, x):
        if self.rf == 1:
            size_out = x.size()[:-1] + (self.nf,)
            x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
            x = x.view(*size_out)
        else:
            raise NotImplementedError
        return x


class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg.n_head == 0
        self.register_buffer('b', torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg.n_head
        self.split_size = n_state
        self.scale = scale
        self.c_attn = Conv1D(n_state * 3, 1, nx)
        self.c_proj = Conv1D(n_state, 1, nx)
        self.attn_dropout = nn.Dropout(cfg.attn_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.b + -1e9 * (1 - self.b)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.b[:, :, (w.size(-1)-w.size(-2)):w.size(-1), :w.size(-1)]
        w = w * b + -1e9 * (1 - b)

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, pre_key=None, pre_value=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        # prevent recalculation
        if pre_key is not None and pre_value is not None:
            key = torch.cat([pre_key, key], dim=-1)
            value = torch.cat([pre_value, value], dim=-2)
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        return a, key, value


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg.n_embd
        self.c_fc = Conv1D(n_state, 1, nx)
        self.c_proj = Conv1D(nx, 1, n_state)
        self.act = ACT_FNS[cfg.afn]
        self.dropout = nn.Dropout(cfg.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)


############## pretrained model loader ###########
##################################################
def load_openai_pretrained_model(model, cfg, n_special, dir):
    """
    load the pretrained OPENAI transformer language model parameters
    :param model: Transformer model
    :param cfg:
    :param n_special: the number of special tokens
    :param dir:
    :return:
    """
    n_ctx = cfg.n_ctx
    n_embd = cfg.n_embd
    n_transfer = cfg.n_layer
    # Load weights from TF model
    print("Loading weights...")
    names = json.load(open(dir + 'parameters_names.json'))
    shapes = json.load(open(dir + 'params_shapes.json'))
    offsets = np.cumsum([np.prod(shape) for shape in shapes])
    init_params = [np.load(dir + 'params_{}.npy'.format(n)) for n in range(10)]
    init_params = np.split(np.concatenate(init_params, 0), offsets)[:-1]
    init_params = [param.reshape(shape) for param, shape in zip(init_params, shapes)]
    if n_ctx > 0:
        init_params[0] = init_params[0][:n_ctx]
    if n_special > 0:
        init_params[0] = np.concatenate(
            [init_params[1],
             (np.random.randn(n_special, n_embd) * 0.02).astype(np.float32),
             init_params[0]
             ], 0)
    else:
        init_params[0] = np.concatenate(
            [init_params[1],
             init_params[0]
             ], 0)
    del init_params[1]
    if n_transfer == -1:
        n_transfer = 0
    else:
        n_transfer = 1 + n_transfer * 12
    init_params = [arr.squeeze() for arr in init_params]

    try:
        assert model.embed.weight.shape == init_params[0].shape
    except AssertionError as e:
        e.args += (model.embed.weight.shape, init_params[0].shape)
        raise

    model.embed.weight.data = torch.from_numpy(init_params[0])

    for name, ip in zip(names[1:n_transfer], init_params[1:n_transfer]):
        name = name[6:]  # skip "model/"
        assert name[-2:] == ":0"
        name = name[:-2]
        name = name.split('/')
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+\d+', m_name):
                l = re.split(r'(\d+)', m_name)
            else:
                l = [m_name]
            pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        try:
            assert pointer.shape == ip.shape
        except AssertionError as e:
            e.args += (pointer.shape, ip.shape)
            raise
        pointer.data = torch.from_numpy(ip)
