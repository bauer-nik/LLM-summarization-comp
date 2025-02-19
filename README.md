# LLM-summarization-comp
The purpose of this project is to explore the novel architecture of Deep Seek: loading and running the R1 model, 
and comparing its performance of a zero-shot summarization task against other available models suited for this task.

The two Deep Seek models loaded and tested are:
* DeepSeek-R1-Distill-Llama-8B
* DeepSeek-R1-Distill-Qwen-32B

Deep Seek summaries were compared against:
* T5
* GPT2
* Baseline - first 5 sentences

## Set up
This project utilizes the transformers library to load the T5 and GPT2 models, as well as torch for cuda access.
Deep Seek is not integrated with the transformers library at this time. Please see DeepSeek-specific Notes to run this experiment with Deep Seek.

### Environment Set-up
Setting up the environment for this project:

Create conda environment with necessary packages and cuda access:
```bash
conda env create -f environment.yaml

conda activate summarization-project

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

### Data / Results Set-up

The data and results directories have been zipped for space. In order to run the summarize script, reader needs to unzip the data.zip.
In order to run the results notebook, reader must unzip results.zip.

Unzipped folders should reflect the following basic folder structure:

    ├── data           # containes 1 csv of test subset
    ├── results        # contains 5 .json files for each experiment run, including baseline


## Experiment Set up
News articles were sourced from Kaggle: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail

A subset of testing data (750 articles) were sampled for this project. That data was zipped for compression.

1. Run summary_eda notebook (optional)
    * This notebook explored average sentence and word lengths of provided article summary to guide
      baseline length and min / max token generation.
2. Run zero shot experiments
   * Main script to run each experiment is: zero_shot_summarize.py
   * Each experiment has its own config file, found under the configs folder
   * Refer DeepSeep-specific Notes to run Deep Seek experiments.
3. Run / explore outputs in results_graphs notebook
    * all experiment outputs, including model summaries, can be found under the results folder after reader has unzipped it
    * If reader is unable to run Deep Seek or other models themselves, they can still explore all experimental outputs in the results_graphs notebook.
    * Outputs were evaluated with Rouge scores for quantitative performance (RougeL, Rouge2, Rouge3), and summaries were sampled and read for qualitative insights / observations
    * Qualitative observations can be found at the bottom of the Readme.

## DeepSeek-specific Notes

The two main issues this project ran into with Deep Seek is: gpu access with the recommended library, and compute resources.
To address the first, the model is loaded inside a flask app, the summary is generated via and endpoint, and the app must be run inside a docker container.
This is integrated in the main script, but requires a bit more manual set-up.
The second problem of compute resources was addressed by accessing remote compute services provided by Lambda Labs. 
It is recommended that the reader have access to GPU(s) with minimum 10 GB total to load in and run the smalled Distilled model.

This project uses the vllm library to work with Distilled Deep Seek models, as recommended by Deep Seek's R1 and V3 documentation.
The vllm library uses the triton package as its means of cuda access, which is only compatible with a Linux OS.
If the reader does not have a Linux OS, a Docker container must be built in order to load and run the Deep Seek models with GPU access.

#### Running Deep Seek Model
1. Build docker container if reader does not have Linux OS
```bash
docker build -t <image_name> .
```
2. run docker if GPU access is sufficient
```bash
docker run --gpus all -p 3050:3050 <image_name>
```
  * --gpus all -> allows gpu access. Can specify GPUs by name if needed.
  * -p 3050:3050 -> maps port 3050 inside docker container to port outside container (needed for flask app)
3. Run app inside docker container
```bash
python3 app.py
```
   * App defaults to llama model. to run qwen model, pass in flag argument
```bash
python3 app.py --ds_version qwen
```

4. Verify IP and Ports match between docker app and zero_shot_summary.
   * defaults to port 3050, and local server 127.0.0.1
   * If reader has different server, update bash variables: DEEPSEEK_SERVER_IP

4. Run zero_shot_summarize.py with Deep Seek config

## Qualitative Observations

A sample of model summaries were read through to evaluate model outputs as they compared to each other and as 
they compared to the reference article summary (highlights). Most model summaries were fairly legible and well structured.
The following model-specific observations were noted from reviewing a sampling of outputs:

### Deep Seek: R1-Distill-Lamma & R1-Distill-Qwen:
* Typically, both Qwen and LLama versions will output a rough summary first before launching into either:
    * a reasoning output (Ok, so ...). It's interesting that this model has anchored on "ok so" / "alright, so" to begin a reasoning section.
    * a type of quiz / prompt output. e.g. the summary will be formatted as listed
  key points (1 key point 2 key point) and then, if given enough time, will prompt itself to write a "quiz" about the key points
    * prompting itself to write the original article (with a word count).
    * re-prompt itself to write the summary
* The summaries aren't too far off from the original highlights, and sometimes surface details that are not in the original summary,
but are in the baseline first 5 sentences (or repeated in another model summary), so it unlikely to be a hallucination.
 
### T5:
* Appears to be the most direct in its language. The highlights given are also very straight-forward as summaries, so the relatively higher T5 performance could be a match of language style, than necessarily a "better summary".
* This model had the most succinct summaries, and is closest to reference article summary.

### GPT2
* Outputs more repeated sentences than the other models tested.
* Possibly an issue with default parameters.

### In Summary
* Most model outputs were better summaries than the rouge scores may suggest. 
  * T5 was the closest to a straight-forward, direct summary. 
  * Deep Seek summaries were better than expected, but the models consistently went past the prompt for a summary and self-prompted with tasks they were assumedly trained on.
  * GPT2 performed the worst out of the box.
* Deep Seeks outputs were intriguing, but might require more resources than a personal project might be able to support
in terms of training or tuning the base models. Overall, the smallest Deep Seek Distilled model outputted decent summaries out of the box.
