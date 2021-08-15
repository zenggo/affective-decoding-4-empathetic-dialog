import torch
from utils import stack_input
import torch.nn.functional as F
import numpy as np


class Generator:
    def __init__(self, model, max_len, indexer, device, role='listener'):
        self.indexer = indexer
        self.max_len = max_len
        self.model = model
        self.device = device
        self.ds_idx = None  # SPEAKER or LISTENER
        self.setRole(role)

    def setRole(self, role):
        if role == 'speaker':
            self.ds_idx = self.indexer.DS_SPEAKER_IDX
        elif role == 'listener':
            self.ds_idx = self.indexer.DS_LISTENER_IDX
        else:
            raise Exception("invalid generator role, should be 'listener' or 'speaker'. ")

    def generate(self, emo, dialog, dialog_state, is_start=True):
        """
        can only respond 1 instance a time
        :param emo:
        :param dialog: 1 x L
        :param dialog_state: 1 x L
        :param is_start: a new sentence or not
        """
        X = stack_input(dialog, [dialog_state], self.indexer)
        X = X.to(self.device)
        if is_start:
            X = self._append_input(X, self.indexer.SOS_IDX)
        with torch.no_grad():
            self.model.eval()
            response = self._generate(emo, X)
        return response

    def _generate(self, emo, X):
        raise NotImplementedError

    def _append_input(self, X, next_token_idx):
        next_pos = X[:, -1:, 1].item() + 1
        next_x = torch.tensor([next_token_idx, next_pos, self.ds_idx]) \
                      .unsqueeze(0).unsqueeze(0)
        return torch.cat((X, next_x.to(self.device)), 1)

    def _cal_next_probs(self, emo, X, token_seq=[], pre_sts=None):
        for token_idx in token_seq:
            X = self._append_input(X, token_idx)
        logits, pre_sts, _ = self.model([emo], X, pre_sts, last_only=True)
        probs = F.log_softmax(logits[0, 0], dim=-1)
        return probs, pre_sts

    def _cal_next_probs_multi(self, emo, X, token_seqs=[], pre_sts_seq=[]):
        assert 0 < len(token_seqs) == len(pre_sts_seq) > 0
        # stack X
        X_seq = []
        for seq in token_seqs:
            tmp = X
            for token_idx in seq:
                tmp = self._append_input(tmp, token_idx)
            X_seq.append(tmp)
        X = torch.cat(X_seq, dim=0)
        # stack previous hidden states
        n = len(token_seqs)
        n_layers = len(pre_sts_seq[0]) - 1
        pre_sts = [sts for sts in pre_sts_seq[0]]
        for i in range(1, n):
            pre_sts[0] = torch.cat([pre_sts[0], pre_sts_seq[i][0]], dim=0)
            for j in range(n_layers):
                h, k, v = pre_sts_seq[i][j+1]
                _h, _k, _v = pre_sts[j+1]
                h = torch.cat([_h, h], dim=0)
                k = torch.cat([_k, k], dim=0)
                v = torch.cat([_v, v], dim=0)
                pre_sts[j+1] = (h, k, v)
        # forward
        logits, pre_sts, _ = self.model([emo for _ in range(X.shape[0])], X, pre_sts, last_only=True)
        probs = F.log_softmax(logits[:, 0], dim=-1)
        # descompose pre_sts
        pre_sts_seq = []
        for i in range(n):
            tmp = []
            tmp.append(pre_sts[0][i:i+1])
            for j in range(n_layers):
                h, k, v = pre_sts[j+1]
                tmp.append((h[i:i+1], k[i:i+1], v[i:i+1]))
            pre_sts_seq.append(tmp)
        # decompose probs - probs.shape: (n, len, n_vocab)
        probs_seq = []
        for i in range(n):
            probs_seq.append(probs[i])
        return probs_seq, pre_sts_seq


    def _token2text(self, token_idxs):
        decoded_text = ""
        for idx in token_idxs:
            if idx >= self.indexer.n_vocab:
                continue
            next_token = self.indexer.decode_index2text(idx)
            if '</w>' in next_token:
                decoded_text += next_token.replace('</w>', '') + ' '
            else:
                decoded_text += next_token
        return decoded_text


class BeamSearchGenerator(Generator):
    def __init__(self, model, max_len, indexer, device, beam_size=5):
        super(BeamSearchGenerator, self).__init__(model, max_len, indexer, device)
        self.beam_size = beam_size

    def _generate(self, emo, X):
        beam_size = self.beam_size
        beams = []  # [ ([idx..], log_prob), ... ]
        dead_beams = []

        next_probs, pre_sts = self._cal_next_probs(emo, X)
        top_probs, top_idxs = next_probs.topk(beam_size)
        for i in range(beam_size):
            beams.append(( [top_idxs[i].item()], top_probs[i].item(), pre_sts ))

        # start beam search
        done = False
        while not done:
            candidates = []
            # batch compute the beams
            seqs = [seq for seq, prob, pre_sts in beams]
            pre_sts_seq = [pre_sts for seq, prob, pre_sts in beams]
            next_probs_seq, pre_sts_seq = self._cal_next_probs_multi(emo, X, seqs, pre_sts_seq)
            # sorting
            for (seq, prob, pre_sts), next_probs, pre_sts in zip(beams, next_probs_seq, pre_sts_seq):
                top_probs, top_idxs = next_probs.topk(beam_size)
                for i in range(beam_size):
                    seq_next = seq + [top_idxs[i].item()]
                    prob_next = prob + top_probs[i].item()
                    candidates.append((seq_next, prob_next, pre_sts))
            candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
            beams = candidates[:beam_size]
            # check dead
            tmp = []
            for seq, prob, pre_sts in beams:
                last = seq[-1]
                if last == self.indexer.EOS_IDX or len(seq) == (self.max_len-1):
                    dead_beams.append((seq, prob))
                    beam_size -= 1
                else:
                    tmp.append((seq, prob, pre_sts))
            beams = tmp
            # check end
            if len(beams) == 0:
                done = True

        texts = [self._token2text(seq) for seq, p in dead_beams]
        # final selection: compared on the normalised log-probabilities
        norm_probs = [p / len(seq) for seq, p in dead_beams]
        final = np.argmax(norm_probs)
        return texts[final], texts
