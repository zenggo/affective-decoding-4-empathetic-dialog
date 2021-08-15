import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from configs import DEFAULT_MODEL_CFG, EMOTION_CATES
from model import ELMModel
from indexer import Indexer
from data_loader import load_dataset
from utils import stack_input, get_time_str, count_parameters, cal_clf_res_detail
from time import time


def parse_args():
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument('--clf_hs', nargs='+', type=int, default=[])
    # other configs
    parser.add_argument('--target_only', default=False, action='store_true')
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='save/memp/b1_std002_h768')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--testid_filter_path', type=str, default='empdial_dataset/testset_idxs_5248.npy')
    parser.add_argument('--testid_sample_path', type=str)
    return parser.parse_args()


def compute_batch_loss(model, batch):
    # stack token, dialog states and position encoding
    X = stack_input(batch['dialog'], [batch['dialog_state']], indexer)
    X = X.to(device)
    # compute augmented LM logits and loss
    lm_logits, _, clf_logits = model(batch['clf_idx'].to(device), X)
    # calculate language modelling loss
    mask = batch['dialog_mask'].to(device)
    if args.target_only:
        for i in range(mask.shape[0]):
            lastspeidx = mask.shape[1] - 1 - batch['dialog_state'][i].tolist()[::-1].index(indexer.DS_SPEAKER_IDX)
            mask[i, :lastspeidx + 2] = 0  # the context and SOS of target are excluded
    target_shifted = X[:, 1:, 0].contiguous().view(-1)
    logits_shifted = lm_logits[:, :-1, :]
    logits_shifted = logits_shifted.contiguous().view(-1, lm_logits.shape[-1])
    lm_loss = F.cross_entropy(logits_shifted, target_shifted, reduction='none')
    mask_shifted = mask[:, 1:]
    lm_loss = torch.sum(lm_loss.view(mask_shifted.shape) * mask_shifted) / torch.sum(mask_shifted)
    # calculate emotion classification loss
    emo_label = batch['emotion']
    clf_loss = F.cross_entropy(clf_logits, emo_label.to(device), reduction='mean')
    # calculate emotion clf accuracy
    clf_res = cal_clf_res_detail(clf_logits, emo_label.tolist())
    return lm_loss.item(), clf_loss.item(), clf_res


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
    cfg.clf_hs = args.clf_hs
    # indexer
    indexer = Indexer(cfg.n_ctx)

    # load test data
    testset, data_loader = load_dataset('test', indexer, batch_size, test=False)
    # test instances whose contexts' length are longer than n_ctx are filtered out
    testset.filter_by_idxs(np.load(args.testid_filter_path))
    if args.testid_sample_path is not None:
        testset.filter_by_idxs(np.load(args.testid_sample_path))

    # load model
    model = ELMModel(cfg, indexer.n_vocab, indexer.n_special, indexer.n_ctx, indexer)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print('Model params: %d' % count_parameters(model))
    model.to(device)


    ################## calculate perplexity on testset ##################
    start_time = time()
    print('Start time: %s' % get_time_str())
    try:
        print('Begin testing PPL.')

        lm_loss = []
        clf_loss = []
        clf_res = [[0, 0, 0] for _ in EMOTION_CATES]
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(data_loader):
                l, c, clfr = compute_batch_loss(model, batch)
                lm_loss.append(l)
                clf_loss.append(c)
                for e in range(len(EMOTION_CATES)):
                    clf_res[e][0] += clfr[e][0]
                    clf_res[e][1] += clfr[e][1]
                    clf_res[e][2] += clfr[e][2]
                if (i+1) % 100 == 0:
                    avg_seconds = (time() - start_time) / (i+1)
                    print('%dth batch, avg time per batch: %f' % (i+1, avg_seconds))

        print('-'*10)
        ppl = np.exp(np.mean(lm_loss))
        print('The perplexity of the model on the testset is: %f' % ppl)
        # calculate accuracy
        acc_c = 0
        acc_top1 = 0
        acc_top5 = 0
        for i in range(len(EMOTION_CATES)):
            print('[%s]: top1_acc = %.3f, top5_acc = %.3f' % \
                  (EMOTION_CATES[i], clf_res[i][1]/clf_res[i][0], clf_res[i][2]/clf_res[i][0]))
            acc_c += clf_res[i][0]
            acc_top1 += clf_res[i][1]
            acc_top5 += clf_res[i][2]
        print('emo classification accuracy top1=%.3f, top5=%.3f' % (acc_top1/acc_c, acc_top5/acc_c))
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from testing early')

    print('Testing end at: %s' % get_time_str())
