import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader

from transformers import T5Tokenizer
from models import  GlossTextCLIP
import utils as utils
from datasets import G2T_Dataset

import os
import time
import argparse, json, datetime
import numpy as np
import yaml
import random
from pathlib import Path
import math
import sys, json
from metrics import bleu, rouge


from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from utils import NativeScaler
from logger import MetricLogger, Logger, ModelLogger, SmoothedValue





def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) scripts', add_help=False)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=80, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)


    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
   
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
    parser.add_argument('--output_dir', default='./outputs',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--config', type=str, required=True)

    return parser

def main(args, config):
    
    if args.eval:
        logger = Logger(log_dir=args.output_dir, prefix='eval')
    else:
        logger = Logger(log_dir=args.output_dir, prefix='train')
    
    utils.init_distributed_mode(args)
    logger.info(json.dumps(vars(args), indent=4))
    

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    device = torch.device(args.device)
    logger.info(f"Device: {device}")
    

    logger.info(f"Creating dataset:")
    tokenizer = T5Tokenizer.from_pretrained(config['model']['tokenizer'])
    
    print(f"Creating model:")
    model = GlossTextCLIP(config=config, task="g2t")
    model = model.to(device)
    logger.info(model)
    
    model_logger = ModelLogger(
        log_dir=args.output_dir, prefix="modellog", 
        model=model, 
        input_size = ((args.batch_size, 128),
                      (args.batch_size, 128),
                      (args.batch_size, 128),
                    (args.batch_size, 128),
        )
    )

    train_data = G2T_Dataset(path=config['data']['path'], tokenizer = tokenizer, config=config, args=args, phase='train')
    logger.info("Train dataset: ")
    logger.info(train_data)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 pin_memory=True,
                                 drop_last=True)
    
    
    
    
    
    dev_data = G2T_Dataset(path=config['data']['path'], tokenizer = tokenizer, config=config, args=args, phase='dev')
    logger.info("Dev dataset: ")
    logger.info(dev_data)
    dev_dataloader = DataLoader(dev_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=dev_data.collate_fn,
                                 pin_memory=True)

    test_data = G2T_Dataset(path=config['data']['path'], tokenizer = tokenizer, config=config, args=args, phase='test')
    logger.info("Test dataset: ")
    logger.info(test_data)
    test_dataloader = DataLoader(test_data,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers, 
                                 collate_fn=test_data.collate_fn,
                                 pin_memory=True)
    


    if args.finetune != '':
        checkpoint = torch.load(args.finetune, map_location='cpu')
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            if "model_text" in key:
                state_dict.pop(key, None)
        ret =  model.load_state_dict(state_dict, strict=False)
        logger.warning('Missing keys: \n' + '\n'.join(ret.missing_keys))
        logger.warning('Unexpected keys: \n' + '\n'.join(ret.unexpected_keys))


    n_parameters = utils.count_parameters_in_MB(model)
    logger.info(f'Number of params: {n_parameters}M')

    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    
    start_epoch = 0
    if args.resume:
        logger.info(f'Resuming training from {args.resume}')
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['model']
        for key in list(state_dict.keys()):
            if "model_text" in key:
                state_dict.pop(key, None)
        ret =  model.load_state_dict(state_dict, strict=False)
        logger.warning('Missing keys: \n' + '\n'.join(ret.missing_keys))
        logger.warning('Unexpected keys: \n' + '\n'.join(ret.unexpected_keys))
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            logger.info(f'Resuming optimizer from {args.resume}')
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        print(f"Performing evaluation on the {len(dev_dataloader)} dev videos and {len(test_dataloader)} test videos")
        if not args.finetune:
            logger.warning('Please specify the trained model: --finetune /path/to/best_checkpoint.pth')
        dev_stats = evaluate(args, dev_dataloader, model, criterion, start_epoch, logger)
        logger.info(f"Dev loss of the network on the {len(dev_dataloader)} test videos: {dev_stats['loss']:.3f}")

        test_stats = evaluate(args, test_dataloader, model, criterion, start_epoch, logger)
        logger.info(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        
        train_stats = evaluate(args, train_dataloader, model, criterion, start_epoch, logger)
        logger.info(f"Train loss of the network on the {len(train_dataloader)} test videos: {train_stats['loss']:.3f}")        
        return

    logger.info(f"Start training from {start_epoch} epochs to {args.epochs} epochs")
    start_time = time.time()
    min_loss = np.inf
    for epoch in range(start_epoch, args.epochs):
        
        train_stats = train_one_epoch(args, model, criterion, 
                              train_dataloader, optimizer, 
                              epoch, loss_scaler, logger, model_logger)

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        dev_stats = evaluate(args, dev_dataloader, model, criterion, epoch, logger)

        if min_loss > dev_stats["loss"]:
            min_loss = dev_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / f'best_checkpoint.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                    }, checkpoint_path)
        
        logger.info(f"* DEV loss {dev_stats['loss']:.3f}, min DEV loss {min_loss}")
       

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'dev_{k}': v for k, v in dev_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        
        logger.write(f"End of Epoch {epoch} with log stats:")
        logger.info(json.dumps(log_stats, indent=4))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    model_logger.save_and_visualize_model(filename=f"{args.output_dir}/model.onnx")
    logger.info('Training time {}'.format(total_time_str))
    
    model_logger.close()

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, logger:Logger, model_logger: ModelLogger):
    model.train(True)

    header = 'Train epoch: [{}/{}]'.format(epoch, args.epochs)
    metric_logger = MetricLogger(delimiter=", ", header=header, logger=logger)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 10

    for step, src_input in enumerate(metric_logger.log_every(data_loader, print_freq)):

        optimizer.zero_grad()
        
        logits = model(src_input=src_input)
        logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_length, vocab_size]
        labels = src_input['labels'].view(-1)  # [batch_size * seq_length]
        loss = criterion(logits, labels.to(logits.device, non_blocking=True))
        loss_scaler(loss, optimizer, epoch, logger=model_logger)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            logger.warning("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        model_logger.log_loss(loss_value, step)
            
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, criterion, epoch, logger: Logger, output_path=None):
    model.eval()

    header = 'Test: '
    metric_logger = MetricLogger(delimiter="  ", header=header, logger=logger)
    print_freq = 10


    with torch.no_grad():
        results = dict()
        for step, src_input in enumerate(metric_logger.log_every(dev_dataloader, print_freq)):

            logits = model(src_input=src_input)
            logits = logits.view(-1, logits.size(-1))  # [batch_size * seq_length, vocab_size]
            labels = src_input['labels'].view(-1)  # [batch_size * seq_length]
            loss = criterion(logits, labels.to(logits.device, non_blocking=True))
            metric_logger.update(loss=loss.item())

            generate_outputs = model.generate(src_input, num_beams=5)
            
            for name, txt_hyp, txt_ref in zip(src_input['name_batch'], 
                                              generate_outputs, 
                                              src_input['text_inputs']):
                
                results[name] = {
                    'hyp' : txt_hyp,
                    'ref' : txt_ref
                }
        
        if output_path is not None:
            with open(output_path, 'w') as f:
                json.dump(results, f)        
        
        txt_hyp = [results[n]['hyp'] for n in results]
        txt_ref = [results[n]['ref'] for n in results]
        bleu_dict = bleu(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
        rouge_score = rouge(references=txt_ref, hypotheses=txt_hyp, level=config['data']['level'])
        for k, v in bleu_dict.items():
            logger.info('{} {:.2f}'.format(k, v))
        logger.info('ROUGE: {:.2f}'.format(rouge_score))
        
        metric_logger.update(bleu1=bleu_dict['bleu1'])
        metric_logger.update(bleu2=bleu_dict['bleu2'])
        metric_logger.update(bleu3=bleu_dict['bleu3'])
        metric_logger.update(bleu4=bleu_dict['bleu4'])
        metric_logger.update(rouge=rouge_score)

        
        
    metric_logger.synchronize_between_processes()
    logger.info(f"* Averaged stats: {metric_logger}")
    logger.info('* DEV loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) scripts', parents=[get_args_parser()])
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, config)