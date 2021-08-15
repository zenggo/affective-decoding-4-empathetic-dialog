import argparse
import random
import numpy as np
import torch
from configs import DEFAULT_MODEL_CFG
from model import ELMModel
from indexer import Indexer
from data_loader import load_dataset
from utils import Logger, count_parameters
from time import time
from generator import BeamSearchGenerator, DBSGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--n_emo_embd', type=int, default=768)
    parser.add_argument('--clf_hs', nargs='+', type=int, default=[])
    parser.add_argument('--tieSL', default=False, action='store_true')
    # generation configs
    parser.add_argument('--max_gen_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--dbs_beam_size', type=int, default=1)
    parser.add_argument('--dbs_groups', type=int, default=5)
    parser.add_argument('--dbs_lambda', type=int, default=0.5)
    # other configs
    parser.add_argument('--model_path', type=str, default='save/elm_1')
    parser.add_argument('--print_to', type=str, default='file')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test_id_filter_path', type=str, default='empdial_dataset/testset_idxs_5248.npy')
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    logger = Logger(args.print_to, args.log_dir, 'testdbs.output')
    logger.log(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model configs
    cfg = DEFAULT_MODEL_CFG
    cfg.n_emo_embd = args.n_emo_embd
    cfg.clf_hs = args.clf_hs
    # indexer
    indexer = Indexer(cfg.n_ctx)

    # load test data
    testset, data_loader = load_dataset('test', indexer, test=True, shuffle=False, batch_size=1)
    # test instances whose contexts' length are longer than 256 are filtered out
    testset.filter_by_idxs(np.load(args.test_id_filter_path))

    model = ELMModel(cfg, indexer.n_vocab, indexer.n_special, indexer.n_ctx, indexer,
                     args.beta, tieSL=args.tieSL)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.log('Model params: %d' % count_parameters(model))
    model.to(device)

    # different decoders
    gen_BS = BeamSearchGenerator(model, args.max_gen_len, indexer, device, args.beam_size)
    gen_DBS = DBSGenerator(model, args.max_gen_len, indexer, device, args.dbs_beam_size, args.dbs_groups, args.dbs_lambda)

    #################### test ####################
    start_time = time()
    _count = 0
    try:
        for (i, b) in enumerate(data_loader):
            logger.log('[Context]:')
            for c in b['data'][0]['context_text']:
                logger.log(' - ' + c)
            logger.log('[Golden]:')
            golden = b['data'][0]['target_text']
            logger.log(' - ' + golden)
            # beam search
            logger.log('[Beam=%d]:' % args.beam_size)
            _, sols = gen_BS.generate(b['clf_idx'], b['dialog'], b['dialog_state'])
            for r in sols:
                logger.log(r)
            # diverse beam search
            logger.log('[DBS(group=%d, beam=%d, lambda=%.2f)]:' % \
                          (args.dbs_groups, args.dbs_beam_size, args.dbs_lambda))
            _, sols, _ = gen_DBS.generate(b['clf_idx'], b['dialog'], b['dialog_state'])
            for r in sols:
                logger.log(r)
            logger.log('\n')
    except KeyboardInterrupt:
        logger.log('-' * 89)
        logger.log('Exiting from test')

    logger.close()
