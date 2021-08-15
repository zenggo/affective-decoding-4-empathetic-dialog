import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from configs import DEFAULT_MODEL_CFG
from model import LMModel
from indexer import Indexer
from data_loader import load_dataset
from utils import stack_input, get_time_str, count_parameters
from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    # other configs
    parser.add_argument('--target_only', default=False, action='store_true')
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='save/best_params')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--testid_filter_path', type=str, default='empdial_dataset/testset_idxs_5248.npy')
    parser.add_argument('--testid_sample_path', type=str)
    return parser.parse_args()


def compute_batch_loss(model, batch):
    # stack token, dialog states and position encoding
    X = stack_input(batch['dialog'], [batch['dialog_state']], indexer)
    X = X.to(device)
    # compute LM logits and loss
    lm_logits, _ = model(X)
    mask = batch['dialog_mask'].to(device)
    if args.target_only:
        for i in range(mask.shape[0]):
            lastspeidx = mask.shape[1] - 1 - batch['dialog_state'][i].tolist()[::-1].index(indexer.DS_SPEAKER_IDX)
            mask[i, :lastspeidx + 2] = 0  # the context and SOS of target are excluded
    # calculate language modelling loss
    target_shifted = X[:, 1:, 0].contiguous().view(-1)
    lm_logits_shifted = lm_logits[:, :-1, :]
    lm_logits_shifted = lm_logits_shifted.contiguous().view(-1, lm_logits.shape[-1])
    loss = F.cross_entropy(lm_logits_shifted, target_shifted, reduction='none')
    mask_shifted = mask[:, 1:]
    loss = torch.sum(loss.view(mask_shifted.shape) * mask_shifted) / torch.sum(mask_shifted)
    return loss


if __name__ == '__main__':

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # batch size
    batch_size = args.n_batch
    # model configs
    cfg = DEFAULT_MODEL_CFG
    # indexer
    indexer = Indexer(cfg.n_ctx)

    # load test data
    testset, data_loader = load_dataset('test', indexer, batch_size, test=False)
    # test instances whose contexts' length are longer than n_ctx are filtered out
    testset.filter_by_idxs(np.load(args.testid_filter_path))
    if args.testid_sample_path is not None:
        testset.filter_by_idxs(np.load(args.testid_sample_path))

    # load model
    model = LMModel(cfg, indexer.n_vocab, indexer.n_special, indexer.n_ctx)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print('Model params: %d' % count_parameters(model))
    model.to(device)


    ################## calculate perplexity on testset ##################
    start_time = time()
    print('Start time: %s' % get_time_str())
    try:
        print('Begin testing PPL.')

        loss = []
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(data_loader):
                l = compute_batch_loss(model, batch)
                loss.append(l.item())
                if (i+1) % 100 == 0:
                    avg_seconds = (time() - start_time) / (i+1)
                    print('%dth batch, avg time per batch: %f' % (i+1, avg_seconds))

        ppl = np.exp(np.mean(loss))
        print('-'*10)
        print('The perplexity of the model on the testset is: %f' % ppl)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from testing early')

    print('Testing end at: %s' % get_time_str())
