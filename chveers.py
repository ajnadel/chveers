# -*- coding: utf-8 -*-
"""chVeers_finetuned_gpt.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1plTyf0YJHWvqe5Bpj0fSBM71ewX-WHHa
"""
# 2: Imports
#region imports

# !pip install transformers
# !pip install termcolor
# !pip install wandb -qqq
import termcolor
import csv
import io
import random
import math
import urllib3
import pandas as pd
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoConfig

from interrupt_handler import GracefulInterruptHandler

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

import argparse

import nltk
from nltk.translate.bleu_score import sentence_bleu as bleu_score

from tqdm import tqdm, trange
tqdm.pandas()
http = urllib3.PoolManager()

from dataset import WANDB_PROJECT_DATASET, EmailBodyDataset
# from PrefixTuning import Trainer_Prefix
from prefix_tuning import PrefixTuning

try:
  import google.colab
  IN_COLAB = True
except:
  IN_COLAB = False

import wandb

#endregion imports

# 3: Arguments:
#region args
argp = argparse.ArgumentParser()
argp.add_argument('function',
  help="Whether to train or evaluate the model",
  choices=["train", "inference"])
argp.add_argument('variant',
  help="Whether to use finetuning, prefix tuning, or baseline GPT",
  choices=["baseline", "finetune", "prefix-tune"])
VARIANTS = {
  'baseline': 'chveers-baseline-gpt',
  'finetune': 'chveers-finetuned-gpt',
  'prefix-tune': 'chveers-prefix-tuned-gpt',
}
argp.add_argument('-m', '--model',
  dest='wandb_model_checkpoint',
  help="Which model to pull from WandB."
)
argp.add_argument('-d', '--dataset',
  dest='dataset',
  help="Artifact identifier of the training dataset.",
  required=True
)
argp.add_argument('--dataset-version',
  dest='dataset_version',
  default='latest',
  help="Artifact identifier of the training dataset."
)
argp.add_argument('-o', '--offline',
  dest='wandb',
  action="store_false",
  help="Do not report metrics or upload artifacts to WandB."
)
argp.add_argument('-e', '--epochs',
  dest='epochs',
  type=int,
  required=True,
  help="Number of epochs to train for"
)
argp.add_argument('--lr',
  dest='lr',
  type=float,
  default=5e-5,
  help="Learning rate."
)
argp.add_argument('--batch-size',
  dest='batch_size',
  type=int,
  default=1,
  help="Batch size."
)

# argp.add_argument('--inference-data',
#   help="Artifact identifier of the inference/validation dataset."
# )
# argp.add_argument('--working-dir',
#  # help="Artifact identifier of the inference/validation dataset."
# )

args = argp.parse_args()
#endregion args

# 4: Set up Weights and Biases integration
print("Setting up wandb.")
wandb.login()
run = wandb.init(project="chveers-finetuned-gpt", entity="chveers", mode=("online" if args.wandb else "offline"), config=args)

print("Loading dataset....")
# 5: Load data file
print(f'Using dataset: {WANDB_PROJECT_DATASET}/{args.dataset}:{args.dataset_version}')
dataset_artif = run.use_artifact(f'{WANDB_PROJECT_DATASET}/{args.dataset}:{args.dataset_version}')
datadir = dataset_artif.download(root=None)
print(datadir)

train_set = pd.read_csv(f'{datadir}/{args.dataset}_train.csv')
dev_set = pd.read_csv(f'{datadir}/{args.dataset}_dev.csv')


print(f"|Train set| = {len(train_set)}")
print(f"|Dev set| = {len(dev_set)}")

"""#### Initialize the GPT-2 Model from Checkpoint"""

VARIANTS = {
  'baseline': 'chveers-baseline-gpt',
  'finetune': 'chveers-finetuned-gpt',
  'prefix-tune': 'chveers-prefix-tuned-gpt',
}

SELECTED_VARIANT = VARIANTS[args.variant]
print(f"Using variant {SELECTED_VARIANT}....")

# MODEL_LOAD_DIR = WORKING_DIR + 'chveers-finetuned-gpt-model'
# MODEL_SAVE_DIR = WORKING_DIR + f'chveers-finetuned-gpt-model-{int(time.time())}'

"""### Loading our GPT-2 from HuggingFace transformers lib

Set `LOAD_LOCAL_CHECKPOINT` to `True` to load from the `MODEL_LOAD_DIR` directory specified in the cells above. Otherwise will load the default GPT-2 checkpoint provided by HuggingFace. This may take a while.
"""

tokenizer = GPT2Tokenizer.from_pretrained('gpt2') # MODEL_SAVE_DIR

pt_config = AutoConfig.from_pretrained('gpt2')
pt_config.vocab_size = len(tokenizer)
# pt.config.vocab_size =  {
#             # train_weights = data_args.train_embs,
#             # optim_prefix = optim_prefix_bool,
#             # preseqlen = model_args.preseqlen,
#             # prefix_dropout = model_args.prefix_dropout,
#             # '': 
            

if args.wandb_model_checkpoint != None:
  print("Loading from WAndB")
  model_at = run.use_artifact(args.wandb_model_checkpoint)
  model_dir = model_at.download()
  if args.variant == 'prefix-tune':
    gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
    model = PrefixTuning.from_pretrained(model_dir,
                    config=pt_config, model_gpt2=gpt2
                    )
  elif args.variant == 'finetune':
    model = GPT2LMHeadModel.from_pretrained(model_dir)
else:
  print("Loading from HuggingFace checkpoint")
  model_dir = f'./{SELECTED_VARIANT}'
  model = GPT2LMHeadModel.from_pretrained('gpt2')
  if args.variant == 'prefix-tune':
    gpt2 = model
    model = PrefixTuning(pt_config, model_gpt2=gpt2)


if args.variant == 'prefix-tune':
  for param in gpt2.base_model.parameters():
    param.requires_grad = False

print("Loaded model.")

"""Next, we will load our training set (processed above into our `EmailBodyDataset` dataset class."""

finetune_dataset = EmailBodyDataset(train_set)
assert len(finetune_dataset) == len(train_set), "We should maintain all of our training examples."
print(f"There are {len(finetune_dataset)} examples in the training dataset.")

# Accumulated batch size (since GPT2 is so big)
# This section is borrowed whole from François St-Amant's post on Towards Data Science, available at
# https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272

def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None

# This section is adapted from François St-Amant's post on Towards Data Science, available at
# https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272

def train(
    dataset, model, tokenizer,
    gpt2=None,
    batch_size=1, epochs=5, lr=2e-5,
    max_seq_len=768, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="chveers_gpt",
    test_mode=False,save_model_on_epoch=False,
    loss_over_time=[],
    enable_pack_tensor=False,
    starting_epoch=0,
    optimizer_state=None,
    scheduler_state=None,
    initial_loss=0
):
    acc_steps = 100
    device = torch.device("cuda")
    print(f"Moving model to device {device}")
    model = model.to(device)
    model.train()
    if args.variant == 'prefix-tune':
      print(f"Moving gpt2 to device {device}")
      gpt2 = gpt2.to(device)
      gpt2.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    if optimizer_state is not None:
      optimizer.load_state_dict(optimizer_state)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )
    if scheduler_state is not None:
      scheduler.load_state_dict(scheduler_state)

    print("optimizing {} parameters...".format(sum(p.numel() for p in model.parameters())))

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss = initial_loss
    accumulating_batch_count = 0
    input_tensor = None

    epochs_completed = 0

    with GracefulInterruptHandler() as h:
      for epoch in range(starting_epoch, epochs + starting_epoch):
          if h.interrupted:
            print("Ending training early!")
            break
          print(f"Training epoch {epoch}")
          print(loss)
          for idx, entry in tqdm(enumerate(train_dataloader)):
              if enable_pack_tensor:
                (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, max_seq_len)

                if carry_on and idx != len(train_dataloader) - 1:
                    continue
              else:
                input_tensor = entry

              input_tensor = input_tensor.to(device)
              if args.variant == 'finetune':
                outputs = model(input_tensor, labels=input_tensor)
              elif args.variant == 'prefix-tune':
                outputs = model(input_tensor, labels=input_tensor, gpt2_model=gpt2)
              loss = outputs[0]
              loss.backward()

              if (accumulating_batch_count % batch_size) == 0:
                  optimizer.step()
                  scheduler.step()
                  optimizer.zero_grad()
                  model.zero_grad()

              accumulating_batch_count += 1
              input_tensor = None
            
          wandb.log({"loss": loss})
          loss_over_time.append(loss)
          epochs_completed += 1
        # if save_model_on_epoch:
        #     torch.save(
        #         model.state_dict(),
        #         os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
        #     )
    return model, loss_over_time, optimizer, scheduler, epochs_completed

"""### Training the model
When it comes to training the model, we work in sets of 250 epochs. The chVeers_300 dataset has only 300 examples, and some of those are held out for the dev set. Thus our epochs run very quickly, especially with CUDA/a GPU.

We've chosen to run 750 epochs (the following cell, 3 time) for each version of our project.
"""


optimizer_state = None
scheduler_state = None
loss = 0

config = {
  'lr': wandb.config['lr'],
  'batch_size': wandb.config['batch_size'],
  'epochs': wandb.config['epochs'],
  'starting_epoch': 0,
}

if args.wandb_model_checkpoint is not None:
  checkpoint = torch.load(f"{model_dir}/torch_states.pt")
  optimizer_state = checkpoint['optimizer_state_dict']
  scheduler_state = checkpoint['scheduler_state_dict']
  config['starting_epoch'] = checkpoint['epochs']
  loss = checkpoint['loss']

print("Config: {}".format(config))

if args.variant == 'prefix-tune':
  model, loss_over_time, optimizer, scheduler, epochs_completed = train(
      finetune_dataset, model, tokenizer, gpt2=gpt2, **config, enable_pack_tensor=False,
      optimizer_state=optimizer_state, scheduler_state=scheduler_state, initial_loss=loss
  )
elif args.variant == 'finetune':
  model, loss_over_time, optimizer, scheduler, epochs_completed = train(
      finetune_dataset, model, tokenizer, **config, enable_pack_tensor=True,
      optimizer_state=optimizer_state, scheduler_state=scheduler_state, initial_loss=loss,
  )
else:
  # idek
  pass

total_epochs = starting_epoch + epochs_completed

trained_model_artif = wandb.Artifact(
    SELECTED_VARIANT + f'_ep{total_epochs}', type="model",
    description="trained {} (total {} epochs)".format(SELECTED_VARIANT, total_epochs)
)

model.save_pretrained(model_dir)
torch.save({
            'epoch': total_epochs,
            'scheduler_state_dict': scheduler.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_over_time[-1],
            }, f"{model_dir}/torch_states.pt")

if args.wandb:
  trained_model_artif.add_dir(model_dir)
  run.log_artifact(trained_model_artif)

# # torch.save(model, './model_750')
# model.save_pretrained(MODEL_SAVE_DIR)
# tokenizer.save_pretrained(MODEL_SAVE_DIR)

# def generate_suffix(prefix, model, tokenizer, no_repeat_ngram_size=4,
#                     skip_special_tokens=False, temperature=0.7):
#   try:
#     prefix_embs = tokenizer(prefix, return_tensors="pt")
#     # prefix_embs['input_ids'] = prefix_embs['input_ids'].cuda()
#     n_tokens_in_prefix = prefix_embs['input_ids'].shape[1]
#     beam_output = model.generate(**prefix_embs, 
#                                 max_length=n_tokens_in_prefix*2, 
#                                 no_repeat_ngram_size=no_repeat_ngram_size, 
#                                 num_beams=5,
#                                 pad_token_id=tokenizer.eos_token_id,
#                                 # top_p=0.92,
#                                 temperature=temperature,
#                                 early_stopping=False)
#     output_flat = beam_output.squeeze(dim=0)[:-1]
#     whole_enchilada = tokenizer.decode(output_flat, skip_special_tokens=skip_special_tokens)
#     suffix_only = tokenizer.decode(output_flat[n_tokens_in_prefix:], skip_special_tokens=skip_special_tokens)
#     return beam_output, whole_enchilada, suffix_only
#   except Exception as e:
#     print(f"Error on prefix '{prefix}':")
#     print(e)
#     return None, None, None
#   # return whole_enchilada, suffix_only

# # generate_suffix(test)
# # print(len(dev_set))
# # example = dev_set.iloc[11]

# # beam_output, whole_enchilada, suffix_only = generate_suffix(example['Body_prefix'], model, tokenizer)
# # print(f"{example['Body']}\n")
# # print("{}{}".format(whole_enchilada[:-len(suffix_only)], termcolor.colored(suffix_only, 'red')))

# # print(beam_output)

# # print([whole_enchilada])

# # reference = example["Body_suffix_gold"].split()
# # hypothesis = "i can get a glimpse of the marathi version....".split()
# # print(reference, hypothesis)
# # blue = bleu_score(reference, hypothesis)
# # print(f"({blue})\t\n{' '.join(reference)}\t\n{' '.join(hypothesis)}")

# dev_set_with_inference = dev_set.copy()
# dev_set_with_inference['Body_suffix_inferred'] = dev_set_with_inference['Body_prefix'].progress_apply(lambda row: generate_suffix(row, model, tokenizer)[2])
# dev_set_with_inference

# # dev_set_with_inference

# dev_set_with_inference.to_csv('./' + f"dev_set_inferred_ep{total_epochs}.csv")

