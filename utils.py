import numpy as np
import torch
import torch.nn.functional as F
import os
from datetime import datetime
import sys
import re
import subprocess
import tempfile
from sklearn.metrics import accuracy_score


########### common utils ###########
####################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def make_path(f):
    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return f


def delete_file(path):
    if os.path.exists(path):
        os.remove(path)


def get_time_str():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


class Logger:
    def __init__(self, print_to, log_dir=None, log_file=None):
        if print_to == 'sys':
            self.log_file = sys.stdout
            print('(log to console)')
        else:  # file
            log_path = os.path.join(log_dir, log_file)
            make_path(log_path)
            self.log_file = open(log_path, 'w')

    def close(self):
        self.log_file.close()

    def log(self, str):
        print(str, file=self.log_file)


########### training/test utils ###########
###########################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def stack_input(X, states, indexer):
    """
    :param X: shape (n, len)
    :param states: list [ (n, len)... ]
    :param indexer:
    :return: shape (n, len, len(states)+2)
    """
    # position encoding
    t = indexer.n_vocab + indexer.n_special
    pos_enc = np.arange(t, t + X.shape[-1])
    pos_enc = torch.from_numpy(pos_enc)
    pos_enc_ext = torch.empty(X.shape, dtype=torch.long)
    for i in np.arange(X.shape[0]):
        pos_enc_ext[i] = pos_enc
    batch = torch.stack([X, pos_enc_ext], dim=-1)
    # states encodings
    for X_state in states:
        X_state = torch.unsqueeze(X_state, -1)
        batch = torch.cat([batch, X_state], dim=-1)
    return batch


def cal_clf_acc(clf_logits, y):
    p = F.softmax(clf_logits, dim=-1)
    # top1 acc
    y_pred = torch.argmax(p, dim=-1).to('cpu').tolist()
    acc_top1 = accuracy_score(y, y_pred)
    # top5 acc
    _, y_top = torch.topk(p, k=5, dim=-1)
    y_top = y_top.to('cpu')
    acc_top5 = 0
    for i in range(len(y)):
        if y[i] in y_top[i].tolist():
            acc_top5 += 1
    acc_top5 /= len(y)
    return acc_top1, acc_top5


def cal_clf_res_detail(clf_logits, y, n_cates=32):
    # [ [n_all, n_correct_top1, n_correct_top5] ]
    res = [[0, 0, 0] for _ in range(n_cates)]
    p = F.softmax(clf_logits, dim=-1)
    # top1
    y_pred = torch.argmax(p, dim=-1).to('cpu').tolist()
    # top5
    _, y_top = torch.topk(p, k=5, dim=-1)
    y_top = y_top.to('cpu')

    for i in range(len(y)):
        res[y[i]][0] += 1
        if y[i] == y_pred[i]:
            res[y[i]][1] += 1
        if y[i] in y_top[i].tolist():
            res[y[i]][2] += 1
    return res


def moses_multi_bleu(hypotheses, references, lowercase=True):
    """Calculate the bleu score for hypotheses and references
    using the MOSES ulti-bleu.perl script.
    Args:
    hypotheses: A numpy array of strings where each string is a single example.
    references: A numpy array of strings where each string is a single example.
    lowercase: If true, pass the "-lc" flag to the multi-bleu script
    Returns:
    The BLEU score as a float32 value.
    """
    if np.size(hypotheses) == 0:
        return np.float32(0.0)

    multi_bleu_path = "./multi-bleu.perl"
    os.chmod(multi_bleu_path, 0o755)

    # Dump hypotheses and references to tempfiles
    hypothesis_file = tempfile.NamedTemporaryFile()
    hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
    hypothesis_file.write(b"\n")
    hypothesis_file.flush()
    reference_file = tempfile.NamedTemporaryFile()
    reference_file.write("\n".join(references).encode("utf-8"))
    reference_file.write(b"\n")
    reference_file.flush()

     # Calculate BLEU using multi-bleu script
    with open(hypothesis_file.name, "r") as read_pred:
        bleu_cmd = [multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference_file.name]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
            bleu_out = bleu_out.decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
            bleu_score = float(bleu_score)
        except subprocess.CalledProcessError as error:
            if error.output is not None:
                print("multi-bleu.perl script returned non-zero exit code")
                print(error.output)
                bleu_score = np.float32(0.0)

    # Close temp files
    hypothesis_file.close()
    reference_file.close()
    return bleu_score