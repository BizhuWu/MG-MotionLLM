import argparse
import torch
import numpy as np
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae
from options import option
import utils.utils_model as utils_model
import json
from dataloader.eval_loader import M2DT_DATALoader
from utils.evaluate import evaluation_m2dt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda



if __name__ == "__main__":

    parser = option.get_args_parser()

    # set hyperparameters
    parser.add_argument("--model_name", type=str, default="./m2dt-ft-from-t5-base/checkpoint-300000/", help="Trained model name or directory")
    parser.add_argument("--prompt", type=str, default="Generate the motion script: ", help="Motion-to-Detailed Text Prompt")
    args = parser.parse_args()



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
    val_loader = M2DT_DATALoader(args.dataname, 'test', 32)



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
    logger = utils_model.get_test_logger(args.model_name, 'test_m2dt_run.log')
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



    # motion aware language model setting
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)



    # start evaluate
    # sequence-level
    bleu1 = []
    bleu4 = []
    bleu7 = []
    rouge = []
    cider = []
    bert_score = []

    # snippet-level
    s_bleu1 = []
    s_bleu4 = []
    s_bleu7 = []
    s_rouge = []
    s_cider = []
    s_bert_score = []
    repeat_time = 1             # follow TM2T to eval once (https://github.com/EricGuo5513/TM2T/blob/main/final_evaluations_m2t.py#L268)

    for _ in range(repeat_time):
        # sequence & snippet levels evaluation
        best_bleu1, best_bleu4, best_bleu7, best_rouge, best_cider, best_bert_score, \
        best_s_bleu1, best_s_bleu4, best_s_bleu7, best_s_rouge, best_s_cider, best_s_bert_score, \
        logger = evaluation_m2dt(
            val_loader,
            vae, model,
            logger,
            tokenizer,
            instruction=args.prompt,
            max_new_tokens=1536
        )
        bleu1.append(best_bleu1)
        bleu4.append(best_bleu4)
        bleu7.append(best_bleu7)
        rouge.append(best_rouge)
        cider.append(best_cider)
        bert_score.append(best_bert_score)

        s_bleu1.append(best_s_bleu1)
        s_bleu4.append(best_s_bleu4)
        s_bleu7.append(best_s_bleu7)
        s_rouge.append(best_s_rouge)
        s_cider.append(best_s_cider)
        s_bert_score.append(best_s_bert_score)

    print('final result:')
    print('sequence-level:')
    print('bleu1: ', sum(bleu1) / repeat_time)
    print('bleu4: ', sum(bleu4) / repeat_time)
    print('bleu7: ', sum(bleu7) / repeat_time)
    print('rouge: ', sum(rouge) / repeat_time)
    print('cider: ', sum(cider) / repeat_time)
    print('bert_score: ', sum(bert_score) / repeat_time)

    print('snippet-level:')
    print('bleu1: ', sum(s_bleu1) / repeat_time)
    print('bleu4: ', sum(s_bleu4) / repeat_time)
    print('bleu7: ', sum(s_bleu7) / repeat_time)
    print('rouge: ', sum(s_rouge) / repeat_time)
    print('cider: ', sum(s_cider) / repeat_time)
    print('bert_score: ', sum(s_bert_score) / repeat_time)

    bleu1 = np.array(bleu1)
    bleu4 = np.array(bleu4)
    bleu7 = np.array(bleu7)
    rouge = np.array(rouge)
    cider = np.array(cider)
    bert_score = np.array(bert_score)

    s_bleu1 = np.array(s_bleu1)
    s_bleu4 = np.array(s_bleu4)
    s_bleu7 = np.array(s_bleu7)
    s_rouge = np.array(s_rouge)
    s_cider = np.array(s_cider)
    s_bert_score = np.array(s_bert_score)

    msg_final = "Sequence-Level:\n"\
                f"bleu1. {np.mean(bleu1):.3f}, conf. {np.std(bleu1) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"bleu4. {np.mean(bleu4):.3f}, conf. {np.std(bleu4) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"bleu7. {np.mean(bleu7):.3f}, conf. {np.std(bleu7) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"rouge. {np.mean(rouge):.3f}, conf. {np.std(rouge) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"cider. {np.mean(cider):.3f}, conf. {np.std(cider) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"bert_score. {np.mean(bert_score):.3f}, conf. {np.std(bert_score) * 1.96 / np.sqrt(repeat_time):.3f}, "
    logger.info(msg_final)

    msg_final = "Snippet-Level:\n"\
                f"bleu1. {np.mean(s_bleu1):.3f}, conf. {np.std(s_bleu1) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"bleu4. {np.mean(s_bleu4):.3f}, conf. {np.std(s_bleu4) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"bleu7. {np.mean(s_bleu7):.3f}, conf. {np.std(s_bleu7) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"rouge. {np.mean(s_rouge):.3f}, conf. {np.std(s_rouge) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"cider. {np.mean(s_cider):.3f}, conf. {np.std(s_cider) * 1.96 / np.sqrt(repeat_time):.3f}, " \
                f"bert_score. {np.mean(s_bert_score):.3f}, conf. {np.std(s_bert_score) * 1.96 / np.sqrt(repeat_time):.3f}, "
    logger.info(msg_final)
