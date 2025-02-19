import torch
import json
from transformers import AutoTokenizer
from src.data_loader import load_df
from src.summarize import get_t5_summary, get_gpt_summary, get_deepseek_summary
from src.utils import load_args, get_metrics, load_model, get_deepseek_model_name


# TODO: ideally refactor config to be passed in via --config flag
# CONFIG_FILE = "configs/t5_zero_shot.json"
# CONFIG_FILE = "configs/gpt2_zero_shot.json"
CONFIG_FILE = "configs/baseline_config.json"
# CONFIG_FILE = "configs/deepseek_zero_shot.json"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_baseline_summary(text):
    if "." in text:
        sentence_list = text.split(".")[:5]
        summary = ".".join(sentence_list)
    else:
        word_list = text.split(" ")[:150]
        summary = " ".join(word_list)

    return summary


def summarize(text, model, tokenizer):
    torch.cuda.empty_cache()

    if args.model_type == "T5":
        return get_t5_summary(text, model, tokenizer, device)

    elif args.model_type == "GPT2":
        return get_gpt_summary(text, model, tokenizer, device)

    elif args.model_type == "Baseline":
        return get_baseline_summary(text)

    elif args.model_type == "DeepSeek":
        summary = get_deepseek_summary(text)
        return summary


if __name__ == '__main__':
    args = load_args(CONFIG_FILE)
    df = load_df()
    # record: which model, keep summary (text), id, metrics
    if args.model_type == "DeepSeek":
        # no mismatch between config and model used
        args.model_name = get_deepseek_model_name()

    test_results = {
        "model": f"{args.model_type} : {args.model_name}",
        "model_summaries": [],
        "article_id": [],
        "rouge_score":[]
    }

    model=None
    tokenizer=None

    if (args.model_type != "Baseline") and (args.model_type != "DeepSeek"):
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = load_model(args).to(device)

    zero_shot_summaries = []

    for _, row in df.iterrows():
        # get zero shot summary
        summary = summarize(row[args.src_col], model, tokenizer)

        zero_shot_summaries.append(summary)
        test_results["model_summaries"].append(summary)
        test_results["article_id"].append(row[args.id_col])

    # get metrics for summary - verify what we actually need.
    results = get_metrics(df[args.tgt_col], zero_shot_summaries)
    test_results.update(results)

    # output results to results/
    with open(f'results/zero_shot_results_{args.model_type}.json', 'w') as fp:
        json.dump(test_results, fp)
