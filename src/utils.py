import argparse
import json
import os
import requests
from collections import defaultdict
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from rouge_score import rouge_scorer


def load_args(config_file):
    parser = argparse.ArgumentParser()

    with open(config_file, "r") as f:
        args_dict = argparse.Namespace()
        args_dict.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=args_dict)

    return args

def load_model(args):
    if args.model_type == "T5":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    elif args.model_type == "GPT2":
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    return model

def get_metrics(tgt_sum, pred_sum):
    rscorer = rouge_scorer.RougeScorer(["rougeL", "rouge2", "rouge3"],use_stemmer=True)
    score_list = [rscorer.score(tgt, pred_sum[x]) for x,tgt in enumerate(tgt_sum)]

    results = defaultdict(list)
    for res_dict in score_list:
        for k,v in res_dict.items():
            results[f"{k}_precision"].append(v.precision)
            results[f"{k}_recall"].append(v.recall)
            results[f"{k}_fmeasure"].append(v.fmeasure)

    return results

def get_deepseek_model_name():
    SERVER_IP = os.getenv("DEEPSEEK_SERVER_IP", "127.0.0.1")
    PORT = os.getenv("DEEPSEEK_PORT", "3050")
    url = f"http://{SERVER_IP}:{PORT}/model"

    response = requests.get(url)
    out = response.text
    return out["model_name"]
