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

    def generate(self, dialog, dialog_state, is_start=True):
        """
        can only respond 1 instance a time
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
            response = self._generate(X)
        return response

    def _generate(self, X):
        raise NotImplementedError

    def _append_input(self, X, next_token_idx):
        next_pos = X[:, -1:, 1].item() + 1
        next_x = torch.tensor([next_token_idx, next_pos, self.ds_idx]) \
                      .unsqueeze(0).unsqueeze(0)
        return torch.cat((X, next_x.to(self.device)), 1)

    def _cal_next_probs(self, X, token_seq=[], pre_sts=None):
        for token_idx in token_seq:
            X = self._append_input(X, token_idx)
        logits, pre_sts = self.model(X, pre_sts, last_only=True)
        probs = F.log_softmax(logits[0, 0], dim=-1)
        return probs, pre_sts

    def _cal_next_probs_multi(self, X, token_seqs=[], pre_sts_seq=[]):
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
        logits, pre_sts = self.model(X, pre_sts, last_only=True)
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


class GreedyGenerator(Generator):
    def __init__(self, model, max_len, indexer, device):
        super(GreedyGenerator, self).__init__(model, max_len, indexer, device)

    def _generate(self, X):
        decoded_idx = []

        pre_sts = None
        for _ in range(self.max_len):
            logits, pre_sts = self.model(X, pre_sts, last_only=True)
            next_idx = torch.argmax(logits[0, 0]).item()
            if next_idx == self.indexer.EOS_IDX:
                break
            decoded_idx.append(next_idx)
            X = self._append_input(X, next_idx)

        return self._token2text(decoded_idx)


class BeamSearchGenerator(Generator):
    def __init__(self, model, max_len, indexer, device, beam_size=5):
        super(BeamSearchGenerator, self).__init__(model, max_len, indexer, device)
        self.beam_size = beam_size

    def _generate(self, X):
        beam_size = self.beam_size
        beams = []  # [ ([idx..], log_prob), ... ]
        dead_beams = []

        next_probs, pre_sts = self._cal_next_probs(X)
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
            next_probs_seq, pre_sts_seq = self._cal_next_probs_multi(X, seqs, pre_sts_seq)
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


class DBSGenerator(Generator):
    """
    Diversity Beam Search: https://arxiv.org/abs/1610.02424
    Hamming Diversity is used to calculate the dissimilarities here.
    """
    def __init__(self, model, max_len, indexer, device,
                 beam_size=1, n_groups=5, lmbd=0.5,
                 reranker='logprob', divfunc='hamming'):
        super(DBSGenerator, self).__init__(model, max_len, indexer, device)
        self.beam_size = beam_size
        self.n_groups = n_groups
        self.lmbd = lmbd
        if reranker == 'logprob':
            self.rerank = self._logprob_rerank
        if divfunc == 'hamming':
            self.aug_div = self._aug_hamming_div

    def _generate(self, X):
        beam_size = self.beam_size
        n_groups = self.n_groups
        groups = []  # [ [([idx..], log_prob), ..], ... ]
        for _ in range(n_groups):
            groups.append([])
        dead_beams = []

        # time=0
        next_probs, pre_sts = self._cal_next_probs(X)
        top_probs, top_idxs = next_probs.topk(beam_size * n_groups)
        for g in range(n_groups):
            for b in range(beam_size):
                i = g * beam_size + b
                groups[g].append(( [top_idxs[i].item()], top_probs[i].item(), pre_sts ))

        # time>0
        done = False
        t = 1
        while not done:
            # batch compute the beams
            seqs = []
            pre_sts_seq = []
            for beams in groups:
                for seq, prob, pre_sts in beams:
                    seqs.append(seq)
                    pre_sts_seq.append(pre_sts)
            next_probs_seq, pre_sts_seq = self._cal_next_probs_multi(X, seqs, pre_sts_seq)
            next_probs_seq.reverse()
            pre_sts_seq.reverse()

            # group 0
            candidates = []
            beam_size_g = len(groups[0])
            for seq, prob, pre_sts in groups[0]:
                next_probs = next_probs_seq.pop()
                pre_sts = pre_sts_seq.pop()
                top_probs, top_idxs = next_probs.topk(beam_size_g)
                for i in range(beam_size_g):
                    seq_next = seq + [top_idxs[i].item()]
                    prob_next = prob + top_probs[i].item()
                    candidates.append((seq_next, prob_next, pre_sts))
            candidates = sorted(candidates, key=lambda c: c[1], reverse=True)
            groups[0] = candidates[:beam_size_g]

            # other groups
            for g in range(1, n_groups):
                candidates = []
                beam_size_g = len(groups[g])
                for seq, prob, pre_sts in groups[g]:
                    next_probs = next_probs_seq.pop()
                    pre_sts = pre_sts_seq.pop()
                    # augment log-probabilities with diversity term
                    probs_aug = self.aug_div(next_probs, t, g, seq, groups, beam_size)
                    # perform one step of beam search for the group
                    top_probs_aug, top_idxs = probs_aug.topk(beam_size_g)
                    for i in range(beam_size_g):
                        token_idx = top_idxs[i].item()
                        seq_next = seq + [token_idx]
                        prob_next = prob + next_probs[token_idx].item()
                        prob_next_aug = prob + top_probs_aug[i].item()
                        candidates.append((seq_next, prob_next, prob_next_aug, pre_sts))
                # sort by the augmented log-prob
                candidates = sorted(candidates, key=lambda c: c[2], reverse=True)
                # but only store the unaugmented log-prob
                groups[g] = [(c[0], c[1], c[3]) for c in candidates[:beam_size_g]]

            # check dead
            for g in range(n_groups):
                tmp = []
                beams = groups[g]
                for b in range(len(beams)):
                    seq, prob, pre_sts = beams[b]
                    last = seq[-1]
                    # if the beam is dead
                    if last == self.indexer.EOS_IDX or len(seq) == (self.max_len - 1):
                        dead_beams.append((seq, prob, g))
                    else:
                        tmp.append((seq, prob, pre_sts))
                groups[g] = tmp
            # check end
            if len(dead_beams) == n_groups*beam_size:
                done = True
            else:
                t += 1

        # final selection: reranking
        sorted_beams = self.rerank(dead_beams)
        texts = [self._token2text(seq) for seq, _, _ in sorted_beams]
        selected_gid = sorted_beams[0][2]
        return texts[0], texts, selected_gid

    # rerankers
    def _logprob_rerank(self, dead_beams):
        # compared on the normalised log-probabilities
        return sorted(dead_beams, key=lambda b: b[1]/len(b[0]), reverse=True)

    # diversity functions
    def _aug_hamming_div(self, probs, t, g, seq, groups, beam_size):
        dissimilarities = torch.zeros(probs.shape).to(self.device)
        for h in range(g):
            for beam in groups[h]:
                token_idx = beam[0][t]
                dissimilarities[token_idx] -= 1 / beam_size
        probs_aug = probs + self.lmbd * dissimilarities
        return probs_aug
