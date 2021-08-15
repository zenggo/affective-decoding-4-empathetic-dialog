import torch
import torch.utils.data as data
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, data, indexer, test=False):
        """
        :param data: { 'context': list, 'target': list  }
        """
        self.contexts = data['context']
        self.targets = data['target']
        self.emotions = data['emotion']
        self.pred_emotions = data['pred_emotion']
        assert len(self.contexts) == len(self.targets) == len(self.emotions)
        self.indexer = indexer
        self.test = test

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        """returns one data pair"""
        item = {}
        item['emotion'] = self.emotions[idx]
        item['pred_emotion'] = self.pred_emotions[idx]

        item['context_text'] = self.contexts[idx]  # dialog utterance list [ str1, str2, ... ]
        encoded_context = self.indexer.encode_text(item['context_text'])  # list [ [wordIdx, ...], ... ]
        context = []
        context_state = []
        for i, c in enumerate(encoded_context):
            context += [self.indexer.SOS_IDX] + c + [self.indexer.EOS_IDX]  # add EOS symbol to every sentence's end
            ds = self.indexer.DS_SPEAKER_IDX if i % 2 == 0 else self.indexer.DS_LISTENER_IDX
            context_state += [ds for _ in range(len(c) + 2)]
        # prepend emotion label to the context
        if self.test:
            encoded_emo = self.indexer.encode_text([item['pred_emotion']])[0]
        else:
            encoded_emo = self.indexer.encode_text([item['emotion']])[0]
        item['context'] = encoded_emo + context
        item['context_state'] = [self.indexer.DS_PREP_IDX for _ in encoded_emo] + context_state

        item['target_text'] = self.targets[idx]  # (str) response
        encoded_target = self.indexer.encode_text([item['target_text']])[0]  # list [wordIdx,...]
        target = [self.indexer.SOS_IDX] + encoded_target + [self.indexer.EOS_IDX]  # add EOS symbol to every sentence's end
        ds = self.indexer.DS_SPEAKER_IDX if len(encoded_context) % 2 == 0 else self.indexer.DS_LISTENER_IDX
        item['target'] = target
        item['target_state'] = [ds for _ in range(len(target))]

        if self.test:
            item['dialog'] = torch.tensor(item['context'], dtype=torch.long)
            item['dialog_state'] = torch.tensor(item['context_state'], dtype=torch.long)
        else:
            item['dialog'] = torch.tensor(item['context'] + item['target'], dtype=torch.long)
            item['dialog_state'] = torch.tensor(item['context_state'] + item['target_state'], dtype=torch.long)
        return item

    def filter_max_len(self, max_len):
        size = len(self.contexts)
        filtered = []
        for i in range(size):
            if self[i]['dialog'].shape[0] <= max_len:
                filtered.append(i)
        self.contexts = self.contexts[filtered]
        self.targets = self.targets[filtered]
        self.emotions = self.emotions[filtered]
        self.pred_emotions = self.pred_emotions[filtered]
        return filtered

    def filter_by_idxs(self, idxs):
        self.contexts = self.contexts[idxs]
        self.targets = self.targets[idxs]
        self.emotions = self.emotions[idxs]
        self.pred_emotions = self.pred_emotions[idxs]


def collate_fn(data, padding_idx):
    """
    merges and padding a list of samples to form a mini-batch
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.empty(len(sequences), max(lengths))
        # no matter what padding_idx is, because mask is used
        padded_seqs = padded_seqs.fill_(padding_idx).long()
        masks = torch.zeros(len(sequences), max(lengths))
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
            masks[i, :end] = 1
        return padded_seqs, lengths, masks

    data.sort(key=lambda x: len(x['dialog']), reverse=True)

    b = {}
    b['data'] = data
    b['emotion'] = [d['emotion'] for d in data]

    dial_batch = [d['dialog'] for d in data]
    dial_state_batch = [d['dialog_state'] for d in data]
    b['dialog'], b['dialog_length'], b['dialog_mask'] = merge(dial_batch)
    b['dialog_state'], _, _ = merge(dial_state_batch)
    return b


def get_data_loader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      collate_fn=lambda data: collate_fn(data, dataset.indexer.PAD_IDX))


def load_dataset(dataset, indexer, batch_size, test=False, shuffle=True):
    d = {}
    d['context'] = np.load('empdial_dataset/sys_dialog_texts.%s.npy' % dataset, allow_pickle=True)
    d['target'] = np.load('empdial_dataset/sys_target_texts.%s.npy' % dataset, allow_pickle=True)
    d['emotion'] = np.load('empdial_dataset/sys_emotion_texts.%s.npy' % dataset, allow_pickle=True)
    d['pred_emotion'] = np.load('empdial_dataset/fasttest_pred_emotion_texts.%s.npy' % dataset, allow_pickle=True)
    dataset = Dataset(d, indexer, test=test)
    data_loader = get_data_loader(dataset, batch_size, shuffle)
    return dataset, data_loader