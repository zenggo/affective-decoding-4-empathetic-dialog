import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from optimizer import OpenAIAdam
from configs import DEFAULT_MODEL_CFG, DEFAULT_OPT_CFG
from model import LMModel, load_openai_pretrained_model
from data_loader import load_dataset
from utils import make_infinite, stack_input, make_path, \
                get_time_str, Logger, delete_file, count_parameters
from time import time
from indexer import Indexer


def parse_args():
    parser = argparse.ArgumentParser()
    # training configs
    parser.add_argument('--n_epoch', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--check_iter', type=int, default=1000)
    parser.add_argument('--print_iter', type=int, default=100)
    parser.add_argument('--max_patience', type=int, default=3)
    # other configs
    parser.add_argument('--log_dir', type=str, default='log/')
    parser.add_argument('--log_file', type=str, default='train.output')
    parser.add_argument('--save_path', type=str, default='save/best_params')
    parser.add_argument('--print_to', type=str, default='file')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_pretrained', default=False, action='store_true')
    parser.add_argument('--pretrained_dir', type=str, default='save/pretrained_lm/')
    return parser.parse_args()


def compute_batch_loss(model, batch):
    # stack token, dialog states and position encoding
    X = stack_input(batch['dialog'], [batch['dialog_state']], indexer)
    X = X.to(device)
    # compute LM logits and loss
    lm_logits, _ = model(X)
    mask = batch['dialog_mask'].to(device)
    # calculate language modelling loss
    target_shifted = X[:, 1:, 0].contiguous().view(-1)
    lm_logits_shifted = lm_logits[:, :-1, :]
    lm_logits_shifted = lm_logits_shifted.contiguous().view(-1, lm_logits.shape[-1])
    loss = F.cross_entropy(lm_logits_shifted, target_shifted, reduction='none')
    mask_shifted = mask[:, 1:]
    loss = torch.sum(loss.view(mask_shifted.shape) * mask_shifted) / torch.sum(mask_shifted)
    return loss


def validate(model, data):
    val_loss = []
    with torch.no_grad():
        model.eval()
        for _, batch in enumerate(data):
            l = compute_batch_loss(model, batch)
            val_loss.append(l.item())
    return np.mean(val_loss)



if __name__ == '__main__':

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # logging
    logger = Logger(args.print_to, args.log_dir, args.log_file)
    logger.log(args)
    # training record by tensorboardX
    tb_writer = SummaryWriter(log_dir=args.log_dir)

    # batch size
    batch_size = args.n_batch
    # model configs
    cfg = DEFAULT_MODEL_CFG
    # indexer
    indexer = Indexer(cfg.n_ctx)

    # load train, dev data
    trainset, data_loader_train = load_dataset('train', indexer, batch_size)
    devset, data_loader_dev = load_dataset('dev', indexer, batch_size)
    # to avoid memory overflow
    trainset.filter_max_len(indexer.n_ctx)
    devset.filter_max_len(indexer.n_ctx)

    # create and load pretrained model
    model = LMModel(cfg, indexer.n_vocab, indexer.n_special, indexer.n_ctx)
    if not args.no_pretrained:
        load_openai_pretrained_model(model.transformer, cfg,
                                     n_special=indexer.n_special,
                                     dir=args.pretrained_dir)

    logger.log('Model params: %d' % count_parameters(model))
    model.to(device)


    #################### training ####################
    tr_iter = make_infinite(data_loader_train)
    n_epoch = args.n_epoch
    n_iter = int(np.ceil(len(trainset) / batch_size)) * n_epoch

    # optimizer
    model_opt = OpenAIAdam(model.parameters(),
                           t_total=n_iter,
                           lr=DEFAULT_OPT_CFG.lr,
                           schedule=DEFAULT_OPT_CFG.lr_schedule,
                           warmup=DEFAULT_OPT_CFG.lr_warmup,
                           b1=DEFAULT_OPT_CFG.b1,
                           b2=DEFAULT_OPT_CFG.b2,
                           e=DEFAULT_OPT_CFG.e,
                           l2=DEFAULT_OPT_CFG.l2,
                           vector_l2=DEFAULT_OPT_CFG.vector_l2,
                           max_grad_norm=DEFAULT_OPT_CFG.max_grad_norm)

    try:
        best_valid_ppl = 1000000
        best_param_path = make_path(args.save_path)
        max_patience = args.max_patience
        patience = max_patience
        check_iter = args.check_iter
        print_iter = args.print_iter

        logger.log('Begin training.')
        logger.log('total training data: %d' % len(trainset))
        logger.log('batch size = %d, epochs: %d, total iterations: %d ' % (batch_size, n_epoch, n_iter))
        logger.log('-'*89)

        start_time = time()
        logger.log('Start time: %s' % get_time_str())
        for i_iter in np.arange(1, n_iter+1):
            batch = next(tr_iter)
            model.train()
            # compute loss
            loss = compute_batch_loss(model, batch)
            # update
            loss.backward()
            model_opt.step()
            model_opt.zero_grad()
            # log
            perplexity = np.exp(min(loss.item(), 100))
            tb_writer.add_scalars('loss', {'loss_train': loss.item()}, i_iter)
            tb_writer.add_scalars('ppl', {'ppl_train': perplexity}, i_iter)
            if i_iter % print_iter == 0:
                tmp = i_iter % check_iter
                tmp = check_iter if tmp == 0 else tmp
                avg_seconds = (time() - start_time) / tmp
                logger.log('iter %d, avg time per iter: %f' % (i_iter, avg_seconds))
            # validate
            if i_iter % check_iter == 0:
                logger.log('-'*10+'start validation at iter %d' % i_iter)
                start_time = time()
                val_loss = validate(model, data_loader_dev)
                val_ppl = np.exp(min(val_loss, 100))
                tb_writer.add_scalars('loss', {'loss_valid': val_loss}, i_iter)
                tb_writer.add_scalars('ppl', {'ppl_valid': val_ppl}, i_iter)
                logger.log('loss=%f, ppl=%f' % (val_loss, val_ppl))
                logger.log('-'*10+'time for validation: %f' % (time()-start_time))
                start_time = time()
                if val_ppl < best_valid_ppl:
                    patience = max_patience
                    best_valid_ppl = val_ppl
                    # save params
                    logger.log('@@ save best params at iter %d, ppl=%.2f' % (i_iter, val_ppl))
                    delete_file(best_param_path)
                    torch.save(model.state_dict(), best_param_path)
                else:
                    patience -= 1
                    if patience == 0:
                        logger.log('-' * 89)
                        logger.log('Exiting from traning at iter %d' % i_iter)
                        break  # break training iteration
    except KeyboardInterrupt:
        logger.log('-' * 89)
        logger.log('Exiting from training early')

    logger.log('Training end at: %s' % get_time_str())
    logger.close()
