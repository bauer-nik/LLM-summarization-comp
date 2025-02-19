import torch
import argparse
from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

app = Flask(__name__)

LLAMA_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
QWEN_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

model = None

def load_model(model_name):
    global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LLM(
            model=model_name,
            task="generate",
            device=device,
            # cpu_offload_gb=3, # offload some compute to CPU if running into CUDA Memory issues
            quantization="fp8",
            max_model_len=3096 # qwen-specifically fails at default max-model-len (131072)
        )


@app.route("/model", methods=['GET'])
def get_model_name():
    # get name of model running
    return jsonify({"model_type": "DeepSeek",
                    "model_name": args.model_name.split("/")[-1]})


@app.route('/summarize', methods=['POST'])
def summarize():
    torch.cuda.empty_cache()
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request"}), 400

    prompt = data.get('prompt')

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, min_tokens=30, max_tokens=120)
    out = model.generate(prompt, sampling_params)
    summary = out[0].outputs[0].text

    return jsonify({"summary": summary})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_version",
                        help="which deepseek version, can be llama or qwen. Model name defers to ds_version",
                        default="llama"
                        )
    parser.add_argument("--model_name", help="the full path to the model used", default=LLAMA_MODEL)
    args = parser.parse_args()

    # TODO: refactor for cleaner consistency
    # verify model name and ds_version match
    if args.ds_version == "llama":
        args.model_name = LLAMA_MODEL
    elif args.ds_version == "qwen":
        args.model_name = QWEN_MODEL
    else:
        raise Exception(f"deepseek version must be either llama or qwen, you typed {args.ds_version}")


    # load specified model version
    load_model(args.model_name)

    # run app
    app.run(host="0.0.0.0", port=3050, debug=False)
