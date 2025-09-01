import argparse, json
import random
import numpy as np
import torch
from os.path import join as pjoin
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from utils.instruction_templates import m2dt_template_list


# Set random seeds and deterministic pytorch for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# Load model and tokenizer
def load_model_and_tokenizer(model_name="google-t5/t5-base"):
    if 'google-t5' in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

        # Add special tokens
        new_tokens = ['<' + str(i) + '>' for i in range(512)]
        new_tokens.extend(['<Motion Tokens>', '</Motion Tokens>', '<Motionless>', '<SEP>'])
        tokenizer.add_tokens(new_tokens)
        model.resize_token_embeddings(len(tokenizer))
    else:
        print('loading ckpt from', model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    return tokenizer, model


# 3. load data
class M2DTDataset(Dataset):
    def __init__(self, tokenizer, split='train', source_len=256, target_len=1536, unit_length=4):
        # t2m
        self.data_root = './dataset/HumanML3D'
        self.text_dir = pjoin(self.data_root, 'texts')
        self.finemotion_text_dir = pjoin(self.data_root, 'finemotion_texts')
        self.split = split
        self.joints_num = 22
        radius = 4
        fps = 20
        dim_pose = 263
        self.unit_length = unit_length

        self.tokenizer = tokenizer
        self.source_len = source_len
        self.target_len = target_len

        # detailed text for motions
        BPMSD_auto_file = pjoin(self.finemotion_text_dir, 'BPMSD_auto.json')
        with open(BPMSD_auto_file, 'r') as f:
            BPMSD_dict = json.load(f)

        BPMSD_human_file = pjoin(self.finemotion_text_dir, 'BPMSD_human.json')
        with open(BPMSD_human_file, 'r') as f:
            BPMSD_human_dict = json.load(f)
        BPMSD_dict.update(BPMSD_human_dict)

        split_file = pjoin(self.data_root, split + '.txt')
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                m_token_list = np.load(pjoin(self.data_root, 'VQVAE_start0', '%s.npy' % name))

                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    line_id = 0
                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['tokens'] = t_tokens

                            if f_tag == 0.0 and to_tag == 0.0:

                                bodyPart_text_list = BPMSD_dict[name]

                                summary_detail_text_dict = text_dict.copy()
                                summary_detail_text_dict['summary'] = caption
                                summary_detail_text_dict['detail'] = bodyPart_text_list
                                text_data.append(summary_detail_text_dict)

                                flag = True

                            else:
                                m_token_list_new = [
                                    tokens[int(f_tag * fps / unit_length): int(to_tag * fps / unit_length)] for tokens
                                    in m_token_list if int(f_tag * fps / unit_length) < int(to_tag * fps / unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue

                                bodyPart_text_list = BPMSD_dict[name][int(f_tag / 0.5): int(to_tag / 0.5)]

                                text_data_new = []

                                # summary + detail
                                summary_detail_text_dict = text_dict.copy()
                                summary_detail_text_dict['summary'] = caption
                                summary_detail_text_dict['detail'] = bodyPart_text_list
                                text_data_new.append(summary_detail_text_dict)

                                new_name = '%s_%f_%f' % (name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'text': text_data_new}
                                new_name_list.append(new_name)
                        except:
                            pass
                        line_id += 1

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'text': text_data}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        """returns the length of dataframe"""
        return len(self.data_dict)

    def __getitem__(self, item):
        """return the input ids, attention masks and target ids"""
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)

        text_data = random.choice(text_list)
        bodyPart_text_list = text_data['detail'][:]

        # Each item in bodyPart_text_list corresponds to 0.5 seconds.
        # Each token corresponds to 4 frames.
        # Here, we ensure strict alignment between motion tokens and detailed body part movement descriptions.
        # (5 tokens = 20 frames = 1 second = 2 items in bodyPart_text_list)
        if m_tokens.shape[0] < 5:
            new_idx = random.randint(0, len(self.data_dict) - 1)
            return self.__getitem__(new_idx)

        m_tokens = m_tokens[:(m_tokens.shape[0]//5) * 5]
        bodyPart_text_list = bodyPart_text_list[:int(m_tokens.shape[0]/2.5)]

        isAug = np.random.choice([True, False])

        if isAug and m_tokens.shape[0] > 5:
            # We augment motions in units of 5 tokens,
            # ensuring strict alignment between motion tokens and detailed body part movement descriptions.
            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

        for i in range(len(bodyPart_text_list)):
            bodyPart_text_item = bodyPart_text_list[i]
            if bodyPart_text_item == "":
                bodyPart_text_list[i] = '<Motionless>'
        long_text = (" <SEP> ").join(bodyPart_text_list)
        detail = long_text

        if self.split == 'train':
            instruction = random.choice(m2dt_template_list)
        else:
            instruction = 'Generate the motion script: <Motion_Placeholder>'

        motion_string = '<Motion Tokens>'
        for token in m_tokens.reshape(-1):
            motion_string += ('<' + str(token) + '>')
        motion_string += '</Motion Tokens>'

        source_text = instruction.replace('<Motion_Placeholder>', motion_string)
        target_text = '### Motion Script ###\n' + detail

        model_inputs = tokenizer(source_text, padding='longest', max_length=self.source_len, truncation=True)
        labels = tokenizer(target_text, padding='longest', max_length=self.target_len, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


if __name__ == "__main__":

    # set hyperparameters
    parser = argparse.ArgumentParser(description="Train on the Motion-to-Detailed Text task.")
    parser.add_argument("--model_name", type=str, default="google-t5/t5-base", help="Pretrained model name or directory")
    parser.add_argument("--output_dir", type=str, default="./m2dt-ft-from-t5-base", help="Directory to save model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Directory to resume model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=300000, help="Max training steps")
    parser.add_argument("--eval_steps", type=int, default=10000, help="Evaluation interval")
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Checkpoint save interval")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_name)

    # load dataset
    print("[Data]: Loading datasets...")
    train_dataset = M2DTDataset(tokenizer, split='train')
    val_dataset = M2DTDataset(tokenizer, split='val')

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # set llm training hyperparameter & trainer
    llm_training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type='cosine',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_total_limit=args.save_total_limit,
        predict_with_generate=True,
        push_to_hub=False,
        report_to=["none"],
    )

    print("[Training]: Starting fine-tuning...\n")
    trainer = Seq2SeqTrainer(
        model=model,
        args=llm_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # start training
    if args.resume_from_checkpoint:
        trainer.train(
            resume_from_checkpoint=args.resume_from_checkpoint
        )
    else:
        trainer.train()
