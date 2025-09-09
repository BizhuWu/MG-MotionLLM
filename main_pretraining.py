import argparse, json
import random
from random import sample
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
from utils.instruction_templates import *


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
class GSPretrainingDataset(Dataset):
    def __init__(self, tokenizer, split='train', source_len=2560, target_len=1536, unit_length=4):
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
                m_token_list = np.load(pjoin(self.data_root, 'VQVAE', '%s.npy' % name))
                m_token_start0_list = np.load(pjoin(self.data_root, 'VQVAE_start0', '%s.npy' % name))

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
                                m_token_start0_list_new = [
                                    tokens[int(f_tag * fps / unit_length): int(to_tag * fps / unit_length)] for tokens
                                    in m_token_start0_list if int(f_tag * fps / unit_length) < int(to_tag * fps / unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                if len(m_token_start0_list) == 0:
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
                                                       'm_token_start0_list': m_token_start0_list_new,
                                                       'text': text_data_new}
                                new_name_list.append(new_name)
                        except:
                            pass
                        line_id += 1

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'm_token_start0_list': m_token_start0_list,
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
        m_token_list, m_token_start0_list, text_list = data['m_token_list'], data['m_token_start0_list'], data['text']
        m_tokens = random.choice(m_token_list)
        m_tokens_start0 = random.choice(m_token_start0_list)

        text_data = random.choice(text_list)
        summary = text_data['summary']
        bodyPart_text_list = text_data['detail'][:]

        # Each item in bodyPart_text_list corresponds to 0.5 seconds.
        # Each token corresponds to 4 frames.
        # Here, we ensure strict alignment between motion tokens and detailed body part movement descriptions.
        # (5 tokens = 20 frames = 1 second = 2 items in bodyPart_text_list)
        m_tokens_start0 = m_tokens_start0[:(m_tokens_start0.shape[0]//5) * 5]
        bodyPart_text_list = bodyPart_text_list[:int(m_tokens_start0.shape[0]/2.5)]

        if m_tokens_start0.shape[0] <= 5 or m_tokens.shape[0] <= 5:
            new_idx = random.randint(0, len(self.data_dict) - 1)
            return self.__getitem__(new_idx)


        source_text = ""
        target_text = ""

        # randomly sampled a task
        random_task = np.random.choice([
            't2m',
            't_headM_2_m',
            't_tailM_2_m',
            't_RandM_2_m',
            'm2t',

            'tdt2m',
            'tdt_headMotion_2_m',
            'tdt_tailMotion_2_m',
            'tdt_RandMotion_2_m',
            'm2tdt',

            'm2dt',
            'dt2Time',
            'Time2dt',
            'm2Time',
            'Time2m',

            'tTime2m',
            't_headMotion_Time_2_m',
            't_tailMotion_Time_2_m',
            't_RandMotion_Time_2_m',
            'tdtTime2m',

            'tdt_headMotion_Time_2_m',
            'tdt_tailMotion_Time_2_m',
            'tdt_RandMotion_Time_2_m',
            'mTime2dt',
            'mdt2Time',

            'tm2Time',
            'tdtm2Time',
            'dtm2Time',
        ])



        if random_task == 't2m':
            coin = np.random.choice([False, False, True])
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:-1]
                else:
                    m_tokens = m_tokens[1:]

            if self.split == 'train':
                instruction = random.choice(t2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate motion: <Caption_Placeholder>'
            source_text = instruction.replace('<Caption_Placeholder>', summary)

            target_text = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 't_headM_2_m':
            coin = np.random.choice([False, False, True])
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:-1]
                else:
                    m_tokens = m_tokens[1:]

            # Input head Motion:
            head_token = m_tokens.reshape(-1)[0]
            condition_string = '<Motion Tokens>' + '<' + str(head_token) + '>' + '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(thm2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate motion start with <Token_Placeholder>: <Caption_Placeholder>'
            source_text = instruction.replace('<Caption_Placeholder>', summary)
            source_text = source_text.replace('<Token_Placeholder>', condition_string)

            target_text = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 't_tailM_2_m':
            coin = np.random.choice([False, False, True])
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:-1]
                else:
                    m_tokens = m_tokens[1:]

            # Input tail Motion:
            tail_token = m_tokens.reshape(-1)[-1]
            condition_string = '<Motion Tokens>' + '<' + str(tail_token) + '>' + '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(ttm2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate motion end with <Token_Placeholder>: <Caption_Placeholder>'
            source_text = instruction.replace('<Caption_Placeholder>', summary)
            source_text = source_text.replace('<Token_Placeholder>', condition_string)

            target_text = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 't_RandM_2_m':
            coin = np.random.choice([False, False, True])
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:-1]
                else:
                    m_tokens = m_tokens[1:]

            # Input Rand Tokens
            if m_tokens.shape[0] < 5:
                sample_num = random.randint(1, m_tokens.shape[0] - 1)
            else:
                sample_num = random.randint(2, 4)
            list_data = m_tokens.reshape(-1).tolist()
            index = sample(list(range(len(list_data))), sample_num)
            index.sort()

            condition_string = '<Motion Tokens>'
            for idx in index:
                token = list_data[idx]
                condition_string += ('<' + str(token) + '>')
            condition_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(trm2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate motion with several key tokens <Tokens_Placeholder>: <Caption_Placeholder>'
            source_text = instruction.replace('<Caption_Placeholder>', summary)
            source_text = source_text.replace('<Tokens_Placeholder>', condition_string)

            target_text = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 'm2t':
            coin = np.random.choice([False, False, True])
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:-1]
                else:
                    m_tokens = m_tokens[1:]

            motion_string = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                motion_string += ('<' + str(token) + '>')
            motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(m2t_template_list)
            else:
                instruction = 'Generate text: <Motion_Placeholder>'
            source_text = instruction.replace('<Motion_Placeholder>', motion_string)

            if random.random() < 0.5:
                summary = f'\"{summary}\"'
            target_text = summary



        if random_task == 'tdt2m':

            isAug = np.random.choice([True, False])

            if isAug:
                # We augment motions in units of 5 tokens,
                # ensuring strict alignment between motion tokens and detailed body part movement descriptions.
                possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
                chosen_idxes = random.sample(possible_idx, 2)
                start_idx = min(chosen_idxes)
                end_idx = max(chosen_idxes)
                m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

                start_text_idx = int(0.4 * start_idx)
                end_text_idx = int(0.4 * end_idx)

                bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]
                bodyPart_text_list = bodyPart_text_list[:int(m_tokens_start0.shape[0]/2.5)]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            if self.split == 'train':
                instruction = random.choice(tdt2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate a motion that matches the motion summary and follows the motion script.'

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            source_text = instruction + '\n\n' + summary + '\n\n'
            source_text = source_text + detail

            target_text = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 'tdt_headMotion_2_m':

            isAug = np.random.choice([True, False])

            if isAug:
                possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
                chosen_idxes = random.sample(possible_idx, 2)
                start_idx = min(chosen_idxes)
                end_idx = max(chosen_idxes)
                m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

                start_text_idx = int(0.4 * start_idx)
                end_text_idx = int(0.4 * end_idx)

                bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            head_token = m_tokens_start0.reshape(-1)[0]
            condition_string = '<Motion Tokens>' + '<' + str(head_token) + '>' + '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tdthm2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate a motion that matches the motion summary and follows the motion script, given the initial <Token_Placeholder>.'
            instruction = instruction.replace('<Token_Placeholder>', condition_string)

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            source_text = instruction + '\n\n' + summary
            source_text = source_text + '\n\n' + detail

            target_text = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 'tdt_tailMotion_2_m':

            isAug = np.random.choice([True, False])

            if isAug:
                possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
                chosen_idxes = random.sample(possible_idx, 2)
                start_idx = min(chosen_idxes)
                end_idx = max(chosen_idxes)
                m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

                start_text_idx = int(0.4 * start_idx)
                end_text_idx = int(0.4 * end_idx)

                bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            tail_token = m_tokens_start0.reshape(-1)[-1]
            condition_string = '<Motion Tokens>' + '<' + str(tail_token) + '>' + '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tdttm2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate a motion that matches the motion summary and follows the motion script, given the last <Token_Placeholder>.'
            instruction = instruction.replace('<Token_Placeholder>', condition_string)

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            source_text = instruction + '\n\n' + summary
            source_text = source_text + '\n\n' + detail

            target_text = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 'tdt_RandMotion_2_m':

            isAug = np.random.choice([True, False])

            if isAug:
                possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
                chosen_idxes = random.sample(possible_idx, 2)
                start_idx = min(chosen_idxes)
                end_idx = max(chosen_idxes)
                m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

                start_text_idx = int(0.4 * start_idx)
                end_text_idx = int(0.4 * end_idx)

                bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            # Input Rand Tokens
            sample_num = random.randint(2, 4)
            list_data = m_tokens_start0.reshape(-1).tolist()
            index = sample(list(range(len(list_data))), sample_num)
            index.sort()

            condition_string = '<Motion Tokens>'
            for idx in index:
                token = list_data[idx]
                condition_string += ('<' + str(token) + '>')
            condition_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tdtrm2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate a motion that matches the motion summary and follows the motion script, given several key tokens <Tokens_Placeholder>.'
            instruction = instruction.replace('<Tokens_Placeholder>', condition_string)

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            source_text = instruction + '\n\n' + summary
            source_text = source_text + '\n\n' + detail

            target_text = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'



        if random_task == 'm2tdt':

            isAug = np.random.choice([True, False])

            if isAug:
                possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
                chosen_idxes = random.sample(possible_idx, 2)
                start_idx = min(chosen_idxes)
                end_idx = max(chosen_idxes)
                m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

                start_text_idx = int(0.4 * start_idx)
                end_text_idx = int(0.4 * end_idx)

                bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            motion_string = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                motion_string += ('<' + str(token) + '>')
            motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(m2tdt_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate the motion summary and the motion script: <Motion_Placeholder>'
            source_text = instruction.replace('<Motion_Placeholder>', motion_string)

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail
            target_text = summary + '\n\n' + detail



        if random_task == 'm2dt':

            isAug = np.random.choice([True, False])

            if isAug:
                # We augment motions in units of 5 tokens,
                # ensuring strict alignment between motion tokens and detailed body part movement descriptions.
                possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
                chosen_idxes = random.sample(possible_idx, 2)
                start_idx = min(chosen_idxes)
                end_idx = max(chosen_idxes)
                m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

                start_text_idx = int(0.4 * start_idx)
                end_text_idx = int(0.4 * end_idx)

                bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            motion_string = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                motion_string += ('<' + str(token) + '>')
            motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(m2dt_template_list)
            else:
                instruction = 'Generate the motion script: <Motion_Placeholder>'
            source_text = instruction.replace('<Motion_Placeholder>', motion_string)

            target_text = '### Motion Script ###\n' + detail



        if random_task == 'dt2Time':

            # Input 1
            possible_idx = list(range(0, len(bodyPart_text_list)))
            chosen_idxes = random.sample(possible_idx, 2)
            start_text_idx = min(chosen_idxes)
            end_text_idx = max(chosen_idxes)

            snippet_bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            original_detail = (" <SEP> ").join(bodyPart_text_list)

            # Input 2
            for i in range(len(snippet_bodyPart_text_list)):
                bodyPart_text_item = snippet_bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    snippet_bodyPart_text_list[i] = '<Motionless>'
            snippet_detail = (" <SEP> ").join(snippet_bodyPart_text_list)

            # Output
            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            target_text = formatted_start_time + ' --> ' + formatted_end_time

            if self.split == 'train':
                instruction = random.choice(dt2time_template_list)
            else:
                instruction = "Determine the start and end times of the snippet of the motion script within the whole motion script."

            snippet_detail = '### Snippet Motion Script ###\n' + snippet_detail
            original_detail = '### Whole Motion Script ###\n' + original_detail

            source_text = instruction + '\n\n' + original_detail + '\n\n'
            source_text = source_text + snippet_detail



        if random_task == 'Time2dt':

            # Input 1
            possible_idx = list(range(0, len(bodyPart_text_list)))
            chosen_idxes = random.sample(possible_idx, 2)
            start_text_idx = min(chosen_idxes)
            end_text_idx = max(chosen_idxes)

            snippet_bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            original_detail = (" <SEP> ").join(bodyPart_text_list)

            # Output
            for i in range(len(snippet_bodyPart_text_list)):
                bodyPart_text_item = snippet_bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    snippet_bodyPart_text_list[i] = '<Motionless>'
            snippet_detail = (" <SEP> ").join(snippet_bodyPart_text_list)

            # Input 2
            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            duration_text = formatted_start_time + ' --> ' + formatted_end_time

            if self.split == 'train':
                instruction = random.choice(time2dt_template_list)
            else:
                instruction = "Output <Duration_Placeholder>'s detail in the whole motion script."
            instruction = instruction.replace('<Duration_Placeholder>', duration_text)

            snippet_detail = "### <Duration_Placeholder>'s Motion Script ###\n" + snippet_detail
            snippet_detail = snippet_detail.replace('<Duration_Placeholder>', duration_text)
            original_detail = '### Whole Motion Script ###\n' + original_detail

            source_text = instruction + '\n\n' + original_detail
            target_text = snippet_detail



        if random_task == 'm2Time':

            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"

            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"

            target_text = formatted_start_time + ' --> ' + formatted_end_time

            whole_motion_string = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                whole_motion_string += ('<' + str(token) + '>')
            whole_motion_string += '</Motion Tokens>'

            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(m2time_template_list)
            else:
                instruction = "Determine the start and end times of <Snippet_Motion_Placeholder> within <Whole_Motion_Placeholder>."
            instruction = instruction.replace('<Snippet_Motion_Placeholder>', snippet_motion_string)
            source_text = instruction.replace('<Whole_Motion_Placeholder>', whole_motion_string)



        if random_task == 'Time2m':

            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"

            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"

            duration_text = formatted_start_time + ' --> ' + formatted_end_time

            whole_motion_string = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                whole_motion_string += ('<' + str(token) + '>')
            whole_motion_string += '</Motion Tokens>'

            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'
            target_text = snippet_motion_string

            if self.split == 'train':
                instruction = random.choice(time2m_template_list)
            else:
                instruction = "Output <Duration_Placeholder>'s motion in <Whole_Motion_Placeholder>."
            instruction = instruction.replace('<Duration_Placeholder>', duration_text)
            source_text = instruction.replace('<Whole_Motion_Placeholder>', whole_motion_string)



        if random_task == 'tTime2m':

            # Input 1: T
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            # Input 2: Time
            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tTime2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet from the motion that corresponds to <Caption_Placeholder>"
            instruction = instruction.replace('<Caption_Placeholder>', summary)
            source_text = instruction.replace('<Duration_Placeholder>', time_text)

            target_text = snippet_motion_string



        if random_task == 't_headMotion_Time_2_m':

            # Input 1: T
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            # Input 2: Time
            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Input 3: head Motion
            head_token = new_m_tokens.reshape(-1)[0]
            condition_string = '<Motion Tokens>' + '<' + str(head_token) + '>' + '</Motion Tokens>'

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tTimehm2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet with initial <Token_Placeholder> from the motion that corresponds to <Caption_Placeholder>"
            instruction = instruction.replace('<Caption_Placeholder>', summary)
            instruction = instruction.replace('<Duration_Placeholder>', time_text)
            source_text = instruction.replace('<Token_Placeholder>', condition_string)

            target_text = snippet_motion_string



        if random_task == 't_tailMotion_Time_2_m':

            # Input 1: T
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            # Input 2: Time
            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Input 3: Tail Motion
            tail_token = new_m_tokens.reshape(-1)[-1]
            condition_string = '<Motion Tokens>' + '<' + str(tail_token) + '>' + '</Motion Tokens>'

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tTimetm2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet with final <Token_Placeholder> from the motion that corresponds to <Caption_Placeholder>"
            instruction = instruction.replace('<Caption_Placeholder>', summary)
            instruction = instruction.replace('<Duration_Placeholder>', time_text)
            source_text = instruction.replace('<Token_Placeholder>', condition_string)

            target_text = snippet_motion_string



        if random_task == 't_RandMotion_Time_2_m':

            # Input 1: T
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            # Input 2: Time
            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Input 3: Rand Motion
            sample_num = random.randint(2, 4)
            list_data = new_m_tokens.reshape(-1).tolist()
            index = sample(list(range(len(list_data))), sample_num)
            index.sort()

            condition_string = '<Motion Tokens>'
            for idx in index:
                token = list_data[idx]
                condition_string += ('<' + str(token) + '>')
            condition_string += '</Motion Tokens>'

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tTimerm2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet with several key tokens <Tokens_Placeholder> from the motion that corresponds to <Caption_Placeholder>"
            instruction = instruction.replace('<Caption_Placeholder>', summary)
            instruction = instruction.replace('<Duration_Placeholder>', time_text)
            source_text = instruction.replace('<Tokens_Placeholder>', condition_string)

            target_text = snippet_motion_string



        if random_task == 'tdtTime2m':

            # Input 1: T+DT
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            text = summary + '\n\n' + detail

            # Input 2: Time
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens_start0.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tdtTime2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet from the motion that matches the motion summary and follows the motion script."
            instruction = instruction.replace('<Duration_Placeholder>', time_text)

            source_text = instruction + '\n\n' + text
            target_text = snippet_motion_string



        if random_task == 'tdt_headMotion_Time_2_m':

            # Input 1: T+DT
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            text = summary + '\n\n' + detail

            # Input 2: Time
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Input 3: head Motion
            head_token = new_m_tokens_start0.reshape(-1)[0]
            condition_string = '<Motion Tokens>' + '<' + str(head_token) + '>' + '</Motion Tokens>'

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens_start0.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tdtTimehm2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet with initial <Token_Placeholder> from the motion that matches the motion summary and follows the motion script."
            instruction = instruction.replace('<Duration_Placeholder>', time_text)
            instruction = instruction.replace('<Token_Placeholder>', condition_string)

            source_text = instruction + '\n\n' + text
            target_text = snippet_motion_string



        if random_task == 'tdt_tailMotion_Time_2_m':

            # Input 1: T+DT
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            text = summary + '\n\n' + detail

            # Input 2: Time
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Input 3: Tail Motion
            tail_token = new_m_tokens_start0.reshape(-1)[-1]
            condition_string = '<Motion Tokens>' + '<' + str(tail_token) + '>' + '</Motion Tokens>'

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens_start0.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tdtTimetm2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet with final <Token_Placeholder> from the motion that matches the motion summary and follows the motion script."
            instruction = instruction.replace('<Duration_Placeholder>', time_text)
            instruction = instruction.replace('<Token_Placeholder>', condition_string)

            source_text = instruction + '\n\n' + text
            target_text = snippet_motion_string



        if random_task == 'tdt_RandMotion_Time_2_m':

            # Input 1: T+DT
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            summary = '### Motion Summary ###\n' + summary
            detail = '### Motion Script ###\n' + detail

            text = summary + '\n\n' + detail

            # Input 2: Time
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            # Input 3: Rand Motion
            list_data = new_m_tokens_start0.reshape(-1).tolist()
            sample_num = random.randint(2, 4)
            index = sample(list(range(len(list_data))), sample_num)
            index.sort()

            condition_string = '<Motion Tokens>'
            for idx in index:
                token = list_data[idx]
                condition_string += ('<' + str(token) + '>')
            condition_string += '</Motion Tokens>'

            # Output: Motion
            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens_start0.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            if self.split == 'train':
                instruction = random.choice(tdtTimerm2m_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion snippet with several key tokens <Tokens_Placeholder> from the motion that matches the motion summary and follows the motion script."

            instruction = instruction.replace('<Duration_Placeholder>', time_text)
            instruction = instruction.replace('<Tokens_Placeholder>', condition_string)

            source_text = instruction + '\n\n' + text
            target_text = snippet_motion_string



        if random_task == 'mTime2dt':

            # Input 1: Time
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            duration_text = formatted_start_time + ' --> ' + formatted_end_time

            # Input 2: Motion
            motion_string = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                motion_string += ('<' + str(token) + '>')
            motion_string += '</Motion Tokens>'

            # Output: detailed text
            snippet_bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]

            for i in range(len(snippet_bodyPart_text_list)):
                bodyPart_text_item = snippet_bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    snippet_bodyPart_text_list[i] = '<Motionless>'
            snippet_detail = (" <SEP> ").join(snippet_bodyPart_text_list)
            snippet_detail = "### <Duration_Placeholder>'s Motion Script ###\n" + snippet_detail
            snippet_detail = snippet_detail.replace('<Duration_Placeholder>', duration_text)

            if self.split == 'train':
                instruction = random.choice(mTime2dt_template_list)
            else:
                instruction = "Generate <Duration_Placeholder>'s motion script for <Motion_Placeholder>"

            source_text = instruction.replace('<Duration_Placeholder>', duration_text)
            source_text = source_text.replace('<Motion_Placeholder>', motion_string)

            target_text = snippet_detail



        if random_task == 'mdt2Time':
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            # Input 1: Motion
            motion_string = '<Motion Tokens>'
            for token in m_tokens_start0.reshape(-1):
                motion_string += ('<' + str(token) + '>')
            motion_string += '</Motion Tokens>'

            # Input 2: DT
            snippet_bodyPart_text_list = bodyPart_text_list[start_text_idx: end_text_idx]  # snippet_bodyPart_text_list 取不到 bodyPart_text_list 的最后一项

            for i in range(len(snippet_bodyPart_text_list)):
                bodyPart_text_item = snippet_bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    snippet_bodyPart_text_list[i] = '<Motionless>'
            snippet_detail = (" <SEP> ").join(snippet_bodyPart_text_list)

            detail = '### Snippet Motion Script ###\n' + snippet_detail

            # Output: Time
            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            if self.split == 'train':
                instruction = random.choice(mdt2Time_template_list)
            else:
                instruction = "Determine the start and end times of the snippet of the motion script within <Motion_Placeholder>."

            source_text = instruction.replace('<Motion_Placeholder>', motion_string)
            source_text = source_text + '\n\n' + detail
            target_text = time_text



        if random_task == 'tm2Time':

            # Input 1: T
            if random.random() < 0.5:
                summary = f'\"{summary}\"'

            # Input 2: Snippet Motion
            possible_idx = list(range(0, m_tokens.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens = m_tokens[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            # Output: Time
            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time

            if self.split == 'train':
                instruction = random.choice(tm2Time_template_list)
            else:
                instruction = "Determine the start and end times of the motion snippet <Tokens_Placeholder> within the whole motion that corresponds to <Caption_Placeholder>"
            instruction = instruction.replace('<Tokens_Placeholder>', snippet_motion_string)
            source_text = instruction.replace('<Caption_Placeholder>', summary)

            target_text = time_text



        if random_task == 'tdtm2Time':

            # Input 1: T + DT
            if random.random() < 0.5:
                summary = f'\"{summary}\"'
            summary = '### Motion Summary ###\n' + summary

            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            detail = '### Motion Script ###\n' + detail


            # Input 2: Snippet Motion
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens_start0.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'


            # Output: Time
            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time


            if self.split == 'train':
                instruction = random.choice(tdtm2Time_template_list)
            else:
                instruction = "Determine the start and end times of the motion snippet <Tokens_Placeholder> within the whole motion that matches the motion summary and follows the motion script."
            source_text = instruction.replace('<Tokens_Placeholder>', snippet_motion_string)
            source_text = source_text + '\n\n' + summary
            source_text = source_text + '\n\n' + detail

            target_text = time_text



        if random_task == 'dtm2Time':

            # Input 1: DT
            for i in range(len(bodyPart_text_list)):
                bodyPart_text_item = bodyPart_text_list[i]
                if bodyPart_text_item == "":
                    bodyPart_text_list[i] = '<Motionless>'
            long_text = (" <SEP> ").join(bodyPart_text_list)
            detail = long_text

            detail = '### Motion Script ###\n' + detail

            # Input 2: Snippet Motion
            possible_idx = list(range(0, m_tokens_start0.shape[0], 5))
            chosen_idxes = random.sample(possible_idx, 2)
            start_idx = min(chosen_idxes)
            end_idx = max(chosen_idxes)
            new_m_tokens_start0 = m_tokens_start0[start_idx:end_idx]

            start_text_idx = int(0.4 * start_idx)
            end_text_idx = int(0.4 * end_idx)

            snippet_motion_string = '<Motion Tokens>'
            for token in new_m_tokens_start0.reshape(-1):
                snippet_motion_string += ('<' + str(token) + '>')
            snippet_motion_string += '</Motion Tokens>'

            # Output: Time
            start_time = start_text_idx * 0.5
            formatted_start_time = f"{start_time:.1f}s"
            end_time = end_text_idx * 0.5
            formatted_end_time = f"{end_time:.1f}s"
            time_text = formatted_start_time + ' --> ' + formatted_end_time


            if self.split == 'train':
                instruction = random.choice(dtm2Time_template_list)
            else:
                instruction = "Determine the start and end times of the motion snippet <Tokens_Placeholder> within the whole motion that follows the motion script."
            source_text = instruction.replace('<Tokens_Placeholder>', snippet_motion_string)
            source_text = source_text + '\n\n' + detail

            target_text = time_text



        # If sampled task fails, turn to the default task: t2m
        if source_text == "" and target_text == "":
            coin = np.random.choice([False, False, True])
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:-1]
                else:
                    m_tokens = m_tokens[1:]

            if self.split == 'train':
                instruction = random.choice(t2m_template_list)
                if random.random() < 0.5:
                    summary = f'\"{summary}\"'
            else:
                instruction = 'Generate motion: <Caption_Placeholder>'
            source_text = instruction.replace('<Caption_Placeholder>', summary)

            target_text = '<Motion Tokens>'
            for token in m_tokens.reshape(-1):
                target_text += ('<' + str(token) + '>')
            target_text += '</Motion Tokens>'

        ###############################################################################

        model_inputs = tokenizer(source_text, padding='longest', max_length=self.source_len, truncation=True)
        labels = tokenizer(target_text, padding='longest', max_length=self.target_len, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


if __name__ == "__main__":

    # set hyperparameters
    parser = argparse.ArgumentParser(description="Granularity-Synergy Pre-training.")
    parser.add_argument("--model_name", type=str, default="google-t5/t5-base",
                        help="Pretrained model name or directory")
    parser.add_argument("--output_dir", type=str, default="./GSPretrained_base", help="Directory to save model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Directory to resume model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Batch size")
    parser.add_argument("--max_steps", type=int, default=300000, help="Max training steps")
    parser.add_argument("--eval_steps", type=int, default=10000, help="Evaluation interval")
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Checkpoint save interval")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(args.model_name)

    # load dataset
    print("[Data]: Loading datasets...")
    train_dataset = GSPretrainingDataset(tokenizer, split='train')
    val_dataset = GSPretrainingDataset(tokenizer, split='val')

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
