import argparse
import torch
import numpy as np
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae
from options import option
import utils.utils_model as utils_model
import json
from dataloader.eval_loader import DATALoader
from utils.evaluate import evaluation
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda



if __name__ == "__main__":

    args = option.get_args_parser()



    # set hyperparameters
    parser = argparse.ArgumentParser(description="Train on the Text-to-Motion task.")
    parser.add_argument("--model_name", type=str, default="./t2m-ft-from-t5-base/checkpoint-300000/", help="Trained model name or directory")
    parser.add_argument("--logger_dir", type=str, default="./t2m-ft-from-t5-base/checkpoint-300000/", help="Directory to save test log")
    parser.add_argument("--prompt", type=str, default="Generate motion: ", help="Text-to-Motion Prompt")
    user_args = parser.parse_args()



    # Evaluator Setting
    if args.dataname == 'kit':
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 21
    else:
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)



    # Test set
    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    val_loader = DATALoader(args.dataname, 'test', 32, w_vectorizer, unit_length=2 ** args.down_t)



    # VQ-VAE
    print('Loading VAE')
    vae = vqvae.HumanVQVAE(args,  ## use args to define different parameters in different quantizers
                           512,
                           args.code_dim,
                           args.output_emb_width,
                           2,
                           args.stride_t,
                           args.width,
                           3,
                           args.dilation_growth_rate)
    resume_pth = f"./checkpoints/pretrained_vqvae/{args.dataname}.pth"
    ckpt = torch.load(resume_pth, map_location='cpu')
    vae.load_state_dict(ckpt['net'], strict=True)
    vae = vae.cuda().eval()
    print('Loading VAE Done')



    # set logger
    logger = utils_model.get_test_logger(user_args.logger_dir, 'test_t2m_run.log')
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



    # motion aware language model setting
    tokenizer = T5Tokenizer.from_pretrained(user_args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(user_args.model_name)

    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)



    # start evaluate
    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    repeat_time = 20

    for _ in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = evaluation(
            val_loader,
            vae, model,
            logger,
            tokenizer,
            eval_wrapper=eval_wrapper,
            instruction=user_args.prompt,
            max_new_tokens=200
        )
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)

    print('final result:')
    print('fid: ', sum(fid) / repeat_time)
    print('div: ', sum(div) / repeat_time)
    print('top1: ', sum(top1) / repeat_time)
    print('top2: ', sum(top2) / repeat_time)
    print('top3: ', sum(top3) / repeat_time)
    print('matching: ', sum(matching) / repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid) * 1.96 / np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div) * 1.96 / np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1) * 1.96 / np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2) * 1.96 / np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3) * 1.96 / np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching) * 1.96 / np.sqrt(repeat_time):.3f}"
    logger.info(msg_final)
