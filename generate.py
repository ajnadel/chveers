import pandas as pd
from transformers import GPT2PreTrainedModel, GPT2Tokenizer
from tqdm import tqdm, trange
tqdm.pandas()

def inference(model: GPT2PreTrainedModel, tokenizer: GPT2Tokenizer, variant: str, df: pd.DataFrame, generate_args={}):
    model = model.cuda()

    if variant == 'prefix-tune':
        assert generate_args['gpt2_model'] is not None, "Missing gpt2 model for prefix-tuning variant"
        generate_args['gpt2_model'] = generate_args['gpt2_model'].cuda()

    assert 'Body_prefix' in df.columns, "missing key 'Body_prefix' in provided dataframe"

    df_with_inference = df.copy()
    print("model before generate_suffix: ", model)
    df_with_inference['Body_suffix_inferred'] = df_with_inference['Body_prefix'].progress_apply(lambda row: generate_suffix(row, model, tokenizer, generate_args=generate_args)[2])

    return df_with_inference

def generate_suffix(prefix, model, tokenizer, no_repeat_ngram_size=4,
                    skip_special_tokens=False, temperature=0.7, generate_args={}):
  try:
    prefix_embs = tokenizer(prefix, return_tensors="pt")
    prefix_embs['input_ids'] = prefix_embs['input_ids'].cuda()
    prefix_embs['attention_mask'] = prefix_embs['attention_mask'].cuda()
    n_tokens_in_prefix = prefix_embs['input_ids'].shape[1]

    beam_output = model.generate(**prefix_embs, 
                                max_length=n_tokens_in_prefix*2, 
                                no_repeat_ngram_size=no_repeat_ngram_size, 
                                num_beams=5,
                                pad_token_id=tokenizer.eos_token_id,
                                # top_p=0.92,
                                temperature=temperature,
                                early_stopping=False, **generate_args, model)
    output_flat = beam_output.squeeze(dim=0)[:-1]
    whole_enchilada = tokenizer.decode(output_flat, skip_special_tokens=skip_special_tokens)
    suffix_only = tokenizer.decode(output_flat[n_tokens_in_prefix:], skip_special_tokens=skip_special_tokens)
    return beam_output, whole_enchilada, suffix_only
  except Exception as e:
    print(f"Error on prefix '{prefix}':")
    print(e)
    return None, None, None