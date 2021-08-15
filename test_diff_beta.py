import argparse
import random
import numpy as np
import torch
from configs import DEFAULT_MODEL_CFG, EMOTION_CATES
from model import ELMModel
from indexer import Indexer
from data_loader import load_dataset
from utils import Logger
from time import time
from generator import BeamSearchGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument('--beta', nargs='+', type=float, default=[])
    parser.add_argument('--n_emo_embd', type=int, default=768)
    # generation configs
    parser.add_argument('--max_gen_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=5)
    # other configs
    parser.add_argument('--n_sample', type=int, default=30)
    parser.add_argument('--model_path', type=str, default='save/elm_1')
    parser.add_argument('--print_to', type=str, default='file')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--log_file', type=str, default='test.output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--testid_filter_path', type=str, default='empdial_dataset/testset_idxs_5248.npy')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = Logger(args.print_to, args.log_dir, args.log_file)
    logger.log(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model configs
    cfg = DEFAULT_MODEL_CFG
    cfg.n_emo_embd = args.n_emo_embd
    # indexer
    indexer = Indexer(cfg.n_ctx)

    # load test data
    testset, data_loader = load_dataset('test', indexer, test=True, shuffle=False, batch_size=1)
    # sample test case
    testset.filter_by_idxs(np.load(args.testid_filter_path))
    sample = np.random.choice(np.load('./save/test_sample_idx.npy'), args.n_sample, replace=False)
    testset.filter_by_idxs(sample)

    # load model
    model = ELMModel(cfg, indexer.n_vocab, indexer.n_special, indexer.n_ctx, indexer, 1.0)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    gen_BS = BeamSearchGenerator(model, args.max_gen_len, indexer, device, args.beam_size)

    # test
    try:
        for (i, b) in enumerate(data_loader):
            logstr = []
            logstr.append('\n[emo]: %s' % b['data'][0]['emotion'])
            logstr.append('[Context]:')
            for c in b['data'][0]['context_text']:
                logstr.append(' - ' + c)
            for beta in args.beta:
                model.set_beta(beta)
                resp, _ = gen_BS.generate(b['emotion'], b['dialog'], b['dialog_state'])
                logstr.append('[beta=%d]: %s' % (beta, resp))
            # logging
            logstr = str.join('\n', logstr)
            logger.log(logstr)

    except KeyboardInterrupt:
        logger.log('-' * 89)
        logger.log('Exiting from test')

    logger.close()

