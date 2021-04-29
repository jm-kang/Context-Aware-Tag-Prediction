# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""


import logging
import math
import os
import json
from copy import deepcopy
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from tqdm.auto import tqdm, trange
import datetime
import neptune
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn
# seaborn.set_context(context="poster")

import sys
sys.path.append('src')

from transformers import (
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    is_torch_tpu_available,
    set_seed,
)

from transformers.modeling_bert import BertConfig, BertForTagGeneration
from transformers.tokenization_bert import BertTokenizer

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    loss_fct: Optional[str] = field(
        default='KL',
        metadata={"help": "Which loss function between KL and CE will be taken"}
    )




@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    
    tag_list: str = field(default="dataset/tagset.txt", metadata={"help": "tag set to consider"})
    tag2contexts: str = field(default="", metadata={"help": "tag to contexts mapping for a diverse inferencing"})

    use_image: bool = field(default=False, metadata={"help": "Whether to use image feature."})
    use_location: bool = field(default=False, metadata={"help": "Whether to use location feature."})
    use_time: bool = field(default=False, metadata={"help": "Whether to use time feature."})
    use_text: bool = field(default=False, metadata={"help": "Whether to use text feature."})



class TrainPost(Dataset):
    def __init__(self, file_path, tokenizer, args, loss_fct):
        self.examples = []
        self.tokenizer = tokenizer
        self.args = args
        self.loss_fct = loss_fct
        self._convert_posts_to_examples(file_path)

    def __getitem__(self, index):
        example = self.examples[index]
        return convert_example_to_feature(example, self.tokenizer, self.args.block_size, self.loss_fct)
        
    def __len__(self):
        return len(self.examples)

    def _convert_posts_to_examples(self, file_path):
        with open(file_path) as f:
            vocab = self.tokenizer.get_vocab()
            json_data = json.load(f)
            for post in tqdm(json_data, desc='posts_to_examples'):
                image = post['azure_caption'] if self.args.use_image else ''
                location = get_semantic_location(post['location']['location_text']) if self.args.use_location else ''
                time = get_semantic_time(post['post']['post_time']) if self.args.use_time else ''
                text = post['post']['post_text'].split('#')[0] if self.args.use_text else ''
                tags = [tag for tag in post['post']['post_tags'] if tag in vocab]

                for i, tag in enumerate(tags):
                    if self.loss_fct == 'KL':
                        self.examples.append({'image': image, 'location': location, 'time': time, 'text': text, 
                                            'previous_tags': tags[:i], 'label': tags[i:]})
                    elif self.loss_fct == 'CE':
                        self.examples.append({'image': image, 'location': location, 'time': time, 'text': text, 
                                            'previous_tags': tags[:i], 'label': tag})

class TestPost(Dataset):
    def __init__(self, file_path, tokenizer, args):
        self.examples = []
        self.tokenizer = tokenizer
        self.args = args
        self._convert_posts_to_examples(file_path)

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _convert_posts_to_examples(self, file_path):
        with open(file_path) as f:
            vocab = self.tokenizer.get_vocab()

            json_data = json.load(f)
            for post in json_data:
                pid = post['id']
                image = post['azure_caption'] if self.args.use_image else ''
                location = get_semantic_location(post['location']['location_text']) if self.args.use_location else ''
                time = get_semantic_time(post['post']['post_time']) if self.args.use_time else ''
                text = post['post']['post_text'].split('#')[0] if self.args.use_text else ''

                tags = [tag for tag in post['post']['post_tags'] if tag in vocab]


                self.examples.append({'pid': pid, 'image': image, 'location': location, 'time': time, 'text': text, 
                                    'previous_tags': [], 'tags': tags})


class DataCollatorForTagGeneration:
    """
    Data collator used for tag generation.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, vocab_size, loss_fct):
        self.vocab_size = vocab_size
        self.loss_fct = loss_fct

    def collate_batch(self, examples):
        if 'labels' in examples[0]:
            if self.loss_fct == 'KL':
                batch_size = len(examples)
                labels = torch.zeros(batch_size, self.vocab_size)
                for i, e in enumerate(examples):
                    labels[i][e['labels']] = 1
            elif self.loss_fct == 'CE':
                labels = torch.tensor([e['labels'] for e in examples], dtype=torch.long)

            input_ids = torch.tensor([e['input_ids'] for e in examples], dtype=torch.long)
            mask_ids = torch.tensor([e['mask_idx'] for e in examples], dtype=torch.long)
            
            return {'input_ids': input_ids, 'labels': labels, 'mask_ids': mask_ids}
        else:
            return examples[0]

def get_semantic_location(raw_location):
    if raw_location == '':
        return '<UNK>'
    return raw_location

def get_semantic_time(raw_time):
    partoftheday = ''
    weekday = ''
    season = ''

    time = datetime.datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%S.000Z').time()
    date = datetime.datetime.strptime(raw_time, '%Y-%m-%dT%H:%M:%S.000Z').date()
    
    hour = time.hour
    if 5 <= hour and hour <= 11: partoftheday = 'morning'
    elif 12 <= hour and hour <= 17: partoftheday = 'afternoon'
    elif 18 <= hour and hour <= 21: partoftheday = 'evening'
    else: partoftheday = 'night'
        
    weekday = date.weekday()
    weekday = 'weekday' if weekday < 5 else 'weekend'
    
    month = date.month
    if 3 <= month and month <= 5: season = 'spring'
    elif 6 <= month and month <= 8: season = 'summer'
    elif 9 <= month and month <= 11: season = 'fall'
    else: season = 'winter'
        
    return partoftheday + ' ' + weekday + ' ' + season

def convert_example_to_feature(example, tokenizer, block_size, loss_fct='KL'):
    image = example['image']
    location = example['location']
    time = example['time']
    text = example['text']
    previous_tags = example['previous_tags']
    label = example['label'] if 'label' in example else None

    image = tokenizer.tokenize(image)
    location = tokenizer.tokenize(location)
    time = tokenizer.tokenize(time)
    
    max_text_size = block_size - len(image) - len(location) - len(time) - len(previous_tags) - 6 # CLS IMG LOC TIME SEP MASK
    text = tokenizer.tokenize(text)[:max_text_size]

    image = ' '.join(image)
    location = ' '.join(location)
    time = ' '.join(time)
    text = ' '.join(text)

    if len(previous_tags) > 0:
        previous_tags = ' '.join(previous_tags)
        _input = tokenizer.cls_token + ' ' + \
                image + ' ' + tokenizer.img_token + ' ' + \
                location + ' ' + tokenizer.loc_token + ' ' + \
                time + ' ' + tokenizer.time_token + ' ' + \
                text + ' ' + tokenizer.sep_token + ' ' + \
                previous_tags + ' ' + tokenizer.mask_token
    else:
        _input = tokenizer.cls_token + ' ' + \
                image + ' ' + tokenizer.img_token + ' ' + \
                location + ' ' + tokenizer.loc_token + ' ' + \
                time + ' ' + tokenizer.time_token + ' ' + \
                text + ' ' + tokenizer.sep_token + ' ' + \
                tokenizer.mask_token
    
    _input_ids = tokenizer.convert_tokens_to_ids(_input.split())
    if label is None: # For evaluation
        return {'pid': example['pid'], 'input_ids': _input_ids}

    # For training
    input_ids = [tokenizer.pad_token_id] * block_size

    input_ids[:len(_input_ids)] = _input_ids
    mask_idx = len(_input_ids) - 1
    if loss_fct == 'KL':
        labels = tokenizer.convert_tokens_to_ids(label) # tag_list
    elif loss_fct == 'CE':
        labels = [-100] * len(input_ids)
        labels[mask_idx] = tokenizer.convert_tokens_to_ids(label)

    return {'input_ids': input_ids, 'labels': labels, 'mask_idx': mask_idx}


def get_dataset(args: DataTrainingArguments, tokenizer: PreTrainedTokenizer, loss_fct='KL', evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if evaluate:
        return TestPost(file_path, tokenizer, args)
    else:
        return TrainPost(file_path, tokenizer, args, loss_fct)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.eval_data_file is None and training_args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.config_name:
        config = BertConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = BertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = BertConfig()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.loss_fct = model_args.loss_fct

    if model_args.tokenizer_name:
        tokenizer = BertTokenizer.from_pretrained(model_args.tokenizer_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        tokenizer = BertTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )

    if model_args.model_name_or_path:
        model = BertForTagGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir
        )
    else:
        logger.info("Training new model from scratch")
        model = BertForTagGeneration.from_config(config)

    # add vocab for special tokens and hashtags
    special_tokens = ['<img>', '<loc>', '<time>']
    num_added_special_toks = tokenizer.add_tokens(special_tokens)
    print('We have added', num_added_special_toks, 'special tokens')
    tokenizer.img_token = '<img>'
    tokenizer.loc_token = '<loc>'
    tokenizer.time_token = '<time>'
    print(tokenizer.convert_tokens_to_ids(special_tokens))
    assert tokenizer.img_token == '<img>'
    assert tokenizer.loc_token == '<loc>'
    assert tokenizer.time_token == '<time>'

    with open(data_args.tag_list) as f:
        tag_list = f.readlines()
        tag_list = ' '.join(tag_list).replace('\n', '').split()
    num_added_toks = tokenizer.add_tokens(tag_list)
    print('tag_list:', data_args.tag_list)
    print('We have added', num_added_toks, 'tokens for hashtags')
    print('total vocab_size:', len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.max_len)

    neptune_project_name = 'junmokang/bertinsta'
    neptune_experiment_name = 'bertinsta-generation'

    if not training_args.do_eval:
        if is_torch_tpu_available():
            if xm.get_ordinal() == 0:
                neptune.init(neptune_project_name)
                neptune.create_experiment(name=neptune_project_name, params=training_args.__dict__)
        else:
            neptune.init(neptune_project_name)
            neptune.create_experiment(name=neptune_project_name, params=training_args.__dict__)

    # Get datasets
    train_dataset = get_dataset(data_args, tokenizer=tokenizer, loss_fct=model_args.loss_fct) if training_args.do_train else None
    eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True) if training_args.do_eval else None
    data_collator = DataCollatorForTagGeneration(config.vocab_size, loss_fct=model_args.loss_fct)

    training_args.per_device_eval_batch_size = 1 # force eval_batch as 1
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        neptune=neptune,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )

    # Training
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        dataloader = trainer.get_eval_dataloader(eval_dataset)
        # multi-gpu eval
        if training_args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        description = "Evaluation"
        batch_size = dataloader.batch_size
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", len(dataloader.dataset))
        logger.info("  Batch size = %d", batch_size)
        model.eval()

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [training_args.device]).per_device_loader(training_args.device)
        
        results = {}
        grouping_results = {}
        # interaction_matrix = np.zeros((6, 6)) # feature interaction
        beam_width = 1
        top_k = 10

        # tag to contexts mapping
        context_list = ['emotion', 'mood', 'location', 'time', 'object', 'activity', 'event', 'others']
        context2ids = {c: [] for c in context_list}
        if data_args.tag2contexts:
            with open(data_args.tag2contexts) as f:
                tag2contexts = json.load(f)
                for tag, contexts in tag2contexts.items():
                    for c in contexts:
                        context2ids[c].append(tag)
                for c in context_list:
                    context2ids[c] = tokenizer.convert_tokens_to_ids(context2ids[c])

        for eid, example in enumerate(tqdm(dataloader, desc=description)):
            generated_tags = beam_decode(beam_width, top_k, model, example, tokenizer, data_args.block_size, training_args.device)
            # generated_tags = beam_decode(beam_width, top_k, model, example, tokenizer, data_args.block_size, training_args.device, None, interaction_matrix) # feature interaction
            results[example['pid']] = generated_tags
            grouping_results[example['pid']] = {}
            grouping_results[example['pid']]['all'] = generated_tags
            # print('all:', str(generated_tags))
        
            # diverse generation (according to context)
            if data_args.tag2contexts:
                for context in context_list:
                    generated_tags = beam_decode(beam_width, top_k, model, example, tokenizer, data_args.block_size, training_args.device, context2ids[context])
                    grouping_results[example['pid']][context] = generated_tags
                    # print(context, ':', str(generated_tags))

        # with np.printoptions(precision=2, suppress=True): # feature interaction
        #         print(interaction_matrix)
        #         print(interaction_matrix.sum(1))
        #         print(interaction_matrix / interaction_matrix.sum(1))

        results_save_path = os.path.join(training_args.output_dir, 'results.json')
        with open(results_save_path, 'w') as f:
            logger.info("saved results.json into %s", training_args.output_dir)
            json.dump(results, f)

        grouping_results_save_path = os.path.join(training_args.output_dir, 'grouping_results.json')
        with open(grouping_results_save_path, 'w') as f:
            logger.info("saved grouping_results.json into %s", training_args.output_dir)
            json.dump(grouping_results, f)

class BeamHypothesis(object):
    def __init__(self, log_prob, previous_tags):
        self.log_prob = log_prob
        self.previous_tags = previous_tags

def beam_decode(beam_width, num_tags, model, example, tokenizer, block_size, device, context2ids=None, interaction_matrix=None):
    hypotheses = []
    orig_vocab_size = 30522
    added_special_toks_size = 3 # <img>, <loc>, <time>

    with torch.no_grad():
        for i in range(num_tags):
            if i == 0:
                feature = convert_example_to_feature(example, tokenizer, block_size)
                input_ids = torch.tensor([feature['input_ids']], dtype=torch.long).to(device)
                inputs = {'input_ids': input_ids}

                outputs = model(**inputs)
                logits = outputs[0]
                
                logit_for_mask = logits[0][-1]    

                if context2ids is not None:
                    context_logit = torch.zeros_like(logit_for_mask)
                    context_logit[context2ids] = 1 
                    logit_for_mask *= context_logit # consider only the corresponding context 

                logit_for_mask[:orig_vocab_size + added_special_toks_size] = -float('inf')

                probabilities = F.log_softmax(logit_for_mask, 0).detach().cpu()

                log_probs, predicted_indices = torch.topk(probabilities, k=beam_width) # deterministic sampling

                predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)

                for log_prob, token in zip(log_probs, predicted_tokens):
                    hypotheses.append(BeamHypothesis(log_prob, [token]))
            else:
                batch_ids = []
                for hypothesis in hypotheses:
                    new_example = {'pid': example['pid'], 'image': example['image'], 'location': example['location'], 'time': example['time'], 'text': example['text'], 'previous_tags': hypothesis.previous_tags}
                    feature = convert_example_to_feature(new_example, tokenizer, block_size)
                    batch_ids.append(feature['input_ids'])
                input_ids = torch.tensor(batch_ids, dtype=torch.long).to(device)
                inputs = {'input_ids': input_ids}

                outputs = model(**inputs, output_attentions=True)

                # if i == num_tags-1: # feature interaction
                #     input_tokens = tokenizer.convert_ids_to_tokens(feature['input_ids'])
                #     attention = outputs[-1][:][0].sum(1) # all layer / first batch
                #     compute_interaction(input_tokens, attention, interaction_matrix)
                #     draw_attention(input_tokens, attention, i)

                logits = outputs[0]

                logits_for_mask = logits[:, -1] # Beam, 31432

                if context2ids is not None:
                    context_logit = torch.zeros_like(logits_for_mask[0])
                    context_logit[context2ids] = 1 
                    logits_for_mask *= context_logit # consider only the corresponding context 

                for hid, hypothesis in enumerate(hypotheses):
                    logits_for_mask[hid, :orig_vocab_size + added_special_toks_size] = -float('inf')
                    previous_tags_ids = tokenizer.convert_tokens_to_ids(hypothesis.previous_tags)
                    logits_for_mask[hid, previous_tags_ids] = -float('inf')

                probabilities = F.log_softmax(logits_for_mask, -1).detach().cpu()
                
                log_probs, predicted_indices = torch.topk(probabilities, k=beam_width) # deterministic sampling

                candidates = []
                for _log_probs, _predicted_indices, hypothesis in zip(log_probs, predicted_indices, hypotheses):
                    _log_probs += hypothesis.log_prob
                    predicted_tokens = tokenizer.convert_ids_to_tokens(_predicted_indices)

                    for log_prob, token in zip(_log_probs, predicted_tokens):
                        candidates.append((log_prob, hypothesis.previous_tags + [token]))

                hypotheses = []
                for log_prob, previous_tags in sorted(candidates, reverse=True)[:beam_width]:
                    hypotheses.append(BeamHypothesis(log_prob, previous_tags))

        return hypotheses[0].previous_tags

def compute_interaction(input_tokens, attention, interaction_matrix):
    cls_idx = input_tokens.index('[CLS]')
    img_idx = input_tokens.index('<img>')
    loc_idx = input_tokens.index('<loc>')
    time_idx = input_tokens.index('<time>')
    sep_idx = input_tokens.index('[SEP]')
    mask_idx = input_tokens.index('[MASK]')

    idx_list = [cls_idx, img_idx, loc_idx, time_idx, sep_idx, mask_idx]

    sum_attention = attention.sum(0) # seq_len X seq_len

    for i in range(5):
        for j in range(5):
            sum_i_to_j = sum_attention[idx_list[i]+1:idx_list[i+1], idx_list[j]+1:idx_list[j+1]].sum().item()
            mean_i_to_j = sum_i_to_j / ((idx_list[i+1] - idx_list[i]+1) * (idx_list[j+1] - idx_list[j]+1))
            interaction_matrix[i][j] += mean_i_to_j

    for j in range(5):
        sum_mask_to_j = sum_attention[mask_idx, idx_list[j]+1:idx_list[j+1]].sum().item()
        mean_mask_to_j = sum_mask_to_j / (idx_list[j+1] - idx_list[j]+1)
        interaction_matrix[-1][j] += mean_mask_to_j

def draw_attention(input_tokens, attention, i):
    fig, axs = plt.subplots(1, 1, figsize=(40, 5))
    
    draw(attention.sum(0).cpu(),
        input_tokens, input_tokens, ax=axs)
    
    plt.savefig('./attention_'+str(i)+'.png')

def draw(data, x, y, ax):
    seaborn.heatmap(data[-1].unsqueeze(0), 
                    xticklabels=x, square=True, yticklabels=[y[-1]], cmap='Reds', vmin=0.0, vmax=1.0, 
                    cbar=False, ax=ax, annot=False, fmt='.1f')


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
