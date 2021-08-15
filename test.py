import argparse
import os
import random
import numpy as np
import torch
from configs import DEFAULT_MODEL_CFG
from model import LMModel
from indexer import Indexer
from data_loader import load_dataset
from utils import get_time_str, Logger, count_parameters, moses_multi_bleu
from time import time
from generator import GreedyGenerator, BeamSearchGenerator, DBSGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    # generation configs
    parser.add_argument('--max_gen_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--dbs_beam_size', type=int, default=1)
    parser.add_argument('--dbs_groups', type=int, default=5)
    parser.add_argument('--dbs_lambda', type=int, default=0.5)
    # other configs
    parser.add_argument('--model_path', type=str, default='save/baseline/transfo')
    parser.add_argument('--print_to', type=str, default='file')
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--log_file', type=str, default='test.output')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--testid_filter_path', type=str, default='empdial_dataset/testset_idxs_5248.npy')
    parser.add_argument('--testid_sample_path', type=str)
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
    # indexer
    indexer = Indexer(cfg.n_ctx)

    # load test data
    testset, data_loader = load_dataset('test', indexer, test=True, shuffle=False, batch_size=1)
    # test instances whose contexts' length are longer than n_ctx are filtered out
    testset.filter_by_idxs(np.load(args.testid_filter_path))
    if args.testid_sample_path is not None:
        testset.filter_by_idxs(np.load(args.testid_sample_path))

    # load model
    model = LMModel(cfg, indexer.n_vocab, indexer.n_special, indexer.n_ctx)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    logger.log('Model params: %d' % count_parameters(model))
    model.to(device)

    # different generators
    # gen_greedy = GreedyGenerator(model, args.max_gen_len, indexer, device)
    gen_BS = BeamSearchGenerator(model, args.max_gen_len, indexer, device, args.beam_size)
    # gen_DBS = DBSGenerator(model, args.max_gen_len, indexer, device, args.dbs_beam_size, args.dbs_groups, args.dbs_lambda)

    #################### test ####################
    start_time = time()
    _count = 0
    try:
        # resp_greedy = []
        resp_beam = []
        # resp_dbs = []
        # dbs_gids = []
        resp_golden = []

        for (i, b) in enumerate(data_loader):
            logstr = []
            logstr.append('[Context]:')
            for c in b['data'][0]['context_text']:
                logstr.append(' - ' + c)
            logstr.append('[Golden]:')
            golden = b['data'][0]['target_text']
            logstr.append(' - ' + golden)
            resp_golden.append(golden)
            # beam search
            logstr.append('[Beam=%d]:' % args.beam_size)
            beam_resp, _ = gen_BS.generate(b['dialog'], b['dialog_state'])
            resp_beam.append(beam_resp)
            logstr.append(' - ' + beam_resp)
            # # diverse beam search
            # logstr.append('[DBS(group=%d, beam=%d, lambda=%.2f)]:' % \
            #               (args.dbs_groups, args.dbs_beam_size, args.dbs_lambda))
            # dbs_resp, _, gid = gen_DBS.generate(b['dialog'], b['dialog_state'])
            # dbs_gids.append(gid)
            # resp_dbs.append(dbs_resp)
            # logstr.append(' - ' + dbs_resp)
            # # greedy decoding
            # logstr.append('[Greedy]:')
            # greedy_resp = gen_greedy.generate(b['dialog'], b['dialog_state'])
            # resp_greedy.append(greedy_resp)
            # logstr.append(' - ' + greedy_resp)
            # logstr.append('=' * 20 + '\n')
            # logging
            logstr = str.join('\n', logstr)
            logger.log(logstr)
            # timing
            _count += 1
            if _count % 10 == 0:
                print('done: %d, avg time: %.2f' % (_count, (time()-start_time)/_count))

        # resp_greedy = np.array(resp_greedy)
        resp_beam = np.array(resp_beam)
        # resp_dbs = np.array(resp_dbs)
        resp_golden = np.array(resp_golden)
        # BLEU
        # bleu_greedy = moses_multi_bleu(resp_greedy, resp_golden, lowercase=True)
        bleu_beam = moses_multi_bleu(resp_beam, resp_golden, lowercase=True)
        # bleu_dbs = moses_multi_bleu(resp_dbs, resp_golden, lowercase=True)
        logger.log('\n' + '-'*10 + '\n')
        logger.log('BLEU beam: %.3f' % bleu_beam)
        # logger.log('BLEU greedy: %.3f, BLEU beam: %.3f, BLEU DBS: %.3f' % \
        #            (bleu_greedy, bleu_beam, bleu_dbs))
        # save outputs
        # np.save(os.path.join(args.log_dir, 'greedy.npy'), resp_greedy)
        np.save(os.path.join(args.log_dir, 'beam.npy'), resp_beam)
        # np.save(os.path.join(args.log_dir, 'dbs.npy'), resp_dbs)
        # np.save(os.path.join(args.log_dir, 'dbs_gids.npy'), dbs_gids)

    except KeyboardInterrupt:
        logger.log('-' * 89)
        logger.log('Exiting from test')

    logger.log('Testing end at: %s, time cost: %.2f' % (get_time_str(), time()-start_time))
    logger.close()
