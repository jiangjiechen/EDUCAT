# EDUCAT

Code for our AAAI 2022 paper (pre-print): "[Unsupervised Editing for Counterfactual Stories](https://arxiv.org/abs/2112.05417)".

## Dependencies

- Prepare requirements: `pip3 install -r requirements.txt`.
- *Set environment variable* `$PJ_HOME`: `export PJ_HOME=/YOUR_PATH/EDUCAT/`.

## Dataset and Models

- We use the [TimeTravel](https://github.com/qkaren/Counterfactual-StoryRW) dataset, as seen in the repo (`$PJ_HOME/data/TimeTravel`). Since EDUCAT is an unsupervised editing method for this task, we do not need training set. 
- Download pre-trained GPT-2 and RoBERTa checkpoints from HuggingFace and put them in `$PJ_HOME/models/`.

## Running EDUCAT

The MCMC sampling part in the code is modified from  [TSMH](https://github.com/Milozms/TSMH) and [CGMH](https://github.com/NingMiao/CGMH). 
See `src/config.py` for parameter details.

- Single-processing version:
```shell
cd $PJ_HOME/src/config.py
python3 counterfactual_rewrite.py \
  --data_path $PJ_HOME/data/TimeTravel/test_data.json \
  --output_file PATH_TO_RESULTS.txt \
  --mlm_path PATH_TO_MODELS/roberta-base/ \
  --gpt2_path PATH_TO_MODELS/gpt2-medium/ \
  --causal_token_finding \
  --coherence_type lm
```

Note that it could be a little time-consuming to directly run EDUCAT (`counterfactual_rewriting.py`). You can use a *pseudo* multi-processing script (`multi_educat.py`) to speed up the editing at the cost of more memory usage. 


- Multi-processing version:
```shell
cd $PJ_HOME/src/config.py
python3 multi_educat.py \
  --data_path $PJ_HOME/data/TimeTravel/test_data.json \
  --output_file PATH_TO_RESULTS.txt \
  --mlm_path PATH_TO_MODELS/roberta-base/ \
  --gpt2_path PATH_TO_MODELS/gpt2-medium/ \
  --causal_token_finding \
  --coherence_type lm \
  -p 8   # 8x speedup
```


## Training EntScore 

EntScore is a regular text classifier. Download the dataset from [TimeTravel](https://github.com/qkaren/Counterfactual-StoryRW), follow the instructions in the script at `src/eval_client/nli_metrics/scripts/`, then you are ready to go:
```shell
cd $PJ_HOME/src/eval_client/nli_metrics/
bash script/base.train.sh 
```

### Pre-trained EntScore

We also provide pre-trained models for EntScore based on the base and large versions of RoBERTa at [Google Drive](https://drive.google.com/file/d/1xGs4C2TPuDK72FoIqmoU-XO0WYAoR4IQ/view?usp=sharing), which are the checkpoints used in the paper. You can download them and put them in `$PJ_HOME/models/nli_metrics/`.

## Evaluation

For evaluating the output using BLEU, BERTScore, EntScore (base and large), go to `src/eval_client` and run:
```shell
cd $PJ_HOME/src/eval_client/
python3 metrics.py \
  --pred_text PATH_TO_PREDICTION \
  --input_json PATH_TO_INPUT \
  --metrics bleu bertscore entailscore 
```

## Citation

If you find our work useful to yours, please kindly cite our paper (pre-print). Formal bibtex will be available after AAAI 2022.
