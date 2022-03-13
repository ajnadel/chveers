import pandas as pd
import os
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
import torch

WANDB_PROJECT_DATASET = 'datasets'

STRAT_WITHHOLD_LAST_N = 0
STRAT_WITHHOLD_LAST_PCT = 1

def datasets_from_csv(filename, *, strategy = 1, withhold_param = 0.5, dev = 30, random_dev=False):
  """Read in and preprocess CSV data from a bytestring. Returns (train_set, dev_set)"""
  parsed_csv = pd.read_csv(filename)
  print(f"Processing {len(parsed_csv)} examples...")

  all_examples = parsed_csv[["Body"]]

  if not random_dev:
    dev_set = all_examples.iloc[:dev]
    train_set = all_examples.iloc[dev:]
  else:
    dev_set = train_set.sample(n = dev)
    train_set = all_examples.loc[~all_examples.index.isin(dev_set.index)]

  train_set = train_set.copy().reset_index(drop=True)
  dev_set = dev_set.copy().reset_index(drop=True)

  if strategy == STRAT_WITHHOLD_LAST_N:
    dev_set['Body_prefix'] = dev_set['Body'].str.split().str[:-withhold_param].apply(' '.join)
    dev_set['Body_suffix_gold'] = dev_set['Body'].str.split().str[-withhold_param:].apply(' '.join)
  elif strategy == STRAT_WITHHOLD_LAST_PCT:
    dev_set['Body_prefix'] = dev_set['Body'].str.split().apply(lambda words: ' '.join(words[:-int(len(words) * withhold_param)]))
    dev_set['Body_suffix_gold'] = dev_set['Body'].str.split().apply(lambda words: ' '.join(words[-int(len(words) * withhold_param):]))
  else:
    raise ValueError('Invalid `strategy` parameter (expected int enum option)')

  return train_set, dev_set


class EmailBodyDataset(Dataset):
  def __init__(self, df, gpt2_type="gpt2", max_length=128, debug=False):
    self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
    
    self.data = []
    for email_body in df["Body"]:
      email_body = email_body.rstrip()
      self.data.append(
        torch.tensor(
          self.tokenizer.encode(
            f"{self.tokenizer.bos_token}{email_body}{self.tokenizer.eos_token}"
          )
        )
      )
      if debug:
        print(f"{self.start_token}{email_body}{self.end_token}\n")
        print(self.data[-1])
        print(len(self.data[-1]))

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
      return self.data[idx]

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser()

  parser.add_argument('dataset_name',
    help="Name token for the dataset.")
  parser.add_argument('csv_filepath',
    help="Path in the local filesystem to a properly-formatted CSV file.")
  strategy_group = parser.add_mutually_exclusive_group(required=True)
  strategy_group.add_argument('-p', '--withhold-pct', type=float,
    help="Withold the last x% of each example for training.")
  strategy_group.add_argument('-n', '--withhold-n', type=int,
    help="Withold the last n words of each example for training.")
  parser.add_argument('-o', '--outfile-prefix',
    help="Path prefix for train and dev dataset output csvs")
  
  parser.add_argument('-d', '--num-dev-examples', type=int, required=True,
    help="The number of examples to reserve for dev (validation) set.")

  parser.add_argument('-u', '--upload', dest="wandb", action="store_true",
    help="Upload resulting datasets to WandB.")

  args = parser.parse_args()

  # start wandb
  if args.wandb:
    import wandb
    run = wandb.init(project=WANDB_PROJECT_DATASET, entity="chveers", job_type='preprocess_data')

  if args.withhold_n is not None:
    strategy = STRAT_WITHHOLD_LAST_N
    withhold_param = args.withhold_n
  else:
    strategy = STRAT_WITHHOLD_LAST_PCT
    withhold_param = args.withhold_pct
  print(f"{args}, p=")

  train_set, dev_set = datasets_from_csv(args.csv_filepath,
                                        strategy=strategy,
                                        withhold_param=withhold_param,
                                        dev=args.num_dev_examples)

  out_prefix = args.outfile_prefix or args.dataset_name
  
  train_out = f"{out_prefix}_train.csv"
  dev_out = f"{out_prefix}_dev.csv"
  print(f"Writing training set to '{train_out}'... ", end="")
  train_set.to_csv(train_out, index=False)
  print('done.')
  print(f"Writing dev set to '{dev_out}'... ", end="")
  dev_set.to_csv(dev_out, index=False)
  print('done.')


  if args.wandb:
    data_artifact = wandb.Artifact(args.dataset_name, type="preprocessed_data")
    data_artifact.add_file(train_out)
    data_artifact.add_file(dev_out)
    run.log_artifact(data_artifact)
