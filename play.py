import argparse
import random
import numpy as np
import torch
from configs import DEFAULT_MODEL_CFG, EMOTION_CATES
from model import ELMModel
from indexer import Indexer
from generator_specemo import BeamSearchGenerator


def parse_args():
    parser = argparse.ArgumentParser()
    # model configs
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--n_emo_embd', type=int, default=768)
    parser.add_argument('--clf_hs', nargs='+', type=int, default=[])
    parser.add_argument('--tieSL', default=False, action='store_true')
    # generation configs
    parser.add_argument('--max_gen_len', type=int, default=50)
    parser.add_argument('--beam_size', type=int, default=5)
    # other configs
    parser.add_argument('--model_path', type=str, default='save/memp/b1_std002_h768')
    parser.add_argument('--print_to', type=str, default='file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--turns', type=int, default=2)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model configs
    cfg = DEFAULT_MODEL_CFG
    cfg.n_emo_embd = args.n_emo_embd
    cfg.clf_hs = args.clf_hs
    # indexer
    indexer = Indexer(cfg.n_ctx)

    # load model
    model = ELMModel(cfg, indexer.n_vocab, indexer.n_special, indexer.n_ctx, indexer,
                     args.beta, tieSL=args.tieSL)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    # different generators
    gen_BS = BeamSearchGenerator(model, args.max_gen_len, indexer, device, args.beam_size)

    #################### play ####################
    while True:
        print('\\Input dialog beginning:')
        starter = input()
        print('\\Input emotion(s):')
        emos = input()

        for emo in emos.split(' '):
            print("# %s" % emo)
            if emo not in EMOTION_CATES:
                print('\\%s not in emotion categories' % emo)
                continue
            else:
                emo = EMOTION_CATES.index(emo)

            ctx = [indexer.SOS_IDX] + indexer.encode_text([starter])[0]
            ds = [indexer.DS_SPEAKER_IDX for _ in range(len(ctx))]

            for i in range(args.turns):
                # speaker
                gen_BS.setRole('speaker')
                utterance, _ = gen_BS.generate(emo, is_start=(False if i==0 else True),
                                dialog=torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(device),
                                dialog_state=torch.tensor(ds, dtype=torch.long).unsqueeze(0).to(device))
                if i == 0:
                    print("- [S]: " + starter + " " + utterance)
                else:
                    print("- [S]: " + utterance)

                ut = indexer.encode_text([utterance])[0] + [indexer.EOS_IDX]
                if i > 0:
                    ut = [indexer.SOS_IDX] + ut
                ctx = ctx + ut
                ds = ds + [indexer.DS_SPEAKER_IDX for _ in range(len(ut))]

                # listener
                gen_BS.setRole('listener')
                utterance, _ = gen_BS.generate(emo, is_start=True,
                                               dialog=torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(device),
                                               dialog_state=torch.tensor(ds, dtype=torch.long).unsqueeze(0).to(device))
                print("- [L]: " + utterance)
                ut = [indexer.SOS_IDX] + indexer.encode_text([utterance])[0] + [indexer.EOS_IDX]
                ctx = ctx + ut
                ds = ds + [indexer.DS_LISTENER_IDX for _ in range(len(ut))]

