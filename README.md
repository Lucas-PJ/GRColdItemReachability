## Project Overview

This is a research codebase focused on the reachability of cold items in generative recommendation. The repository provides scripts for data preprocessing, tokenizer training, semantic ID (SID) generation, and the training of multiple models:

- TIGER
- TIGER_SID
- TIGER_Scorer
- TIGER_Edge
- SASRec

Two tokenizer families are supported:

- RQ-VAE
- OPQ

## Repository Layout

- `data/`: raw datasets and preprocessing scripts
   - `process_data/`: per-dataset preprocessing scripts and processed outputs
- `model/`: TIGER, TIGET-based variants and SASRec model implementations and training scripts
- `opq/`: OPQ tokenizer training and code generation
- `rqvae/`: RQ-VAE tokenizer training and code generation

## Environment

Install the main Python dependencies (adjust versions as needed):

```bash
pip install torch transformers sentence-transformers numpy pandas pyarrow tqdm
```

## Data Preprocessing

1. Download a dataset and place it under `data/`. Supported datasets include:
    - Amazon 5-core (e.g. `Beauty`, `Sports_and_Outdoors`, `Toys_and_Games`). Download from this [link](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html).
    - GDGB datasets (e.g. `Sephora`, `WeiboDaily`, `WeiboTech`, `WikiLife`). Download from this [link](https://www.kaggle.com/datasets/f5e51c13e31f34cc84177d121c5902e0076c826d24e40414186024232e62973e).

    Example layout:

    ```text
    data/Beauty/reviews_Beauty_5.json
    data/Beauty/meta_Beauty.json
    
    data/Sephora/sephora_edge.csv
    data/Sephora/sephora_node.csv
    ```

2. Run the corresponding preprocessing script in `data/process_data/` to produce processed Parquet files and mappings (e.g. `item_emb.parquet`, `filtered_item_id_mapping.npy`):

```bash
python process_sliding_timecut_rawdata_gdgb_sephora_person.py

python generate_filtered_item_id_mapping.py  --item_emb_path ../data/process_data/Sephora_sliding/item_emb.parquet
```


Outputs are saved to `data/process_data/<DatasetName>/`, typically including `item_emb.parquet`, `filtered_item_id_mapping.npy`, `user_mapping.npy`, and `item_mapping.npy`.

## Tokenizer Training & SID Generation

Training a tokenizer and generating semantic IDs produces a file under `data/process_data/<DatasetName>/` such as `<DatasetName>_t5_rqvae.npy` or `<DatasetName>_opq.npy`.


### RQ-VAE

Train RQ-VAE tokenizer:

```bash
cd rqvae
python main.py --dataset <DatasetName>
```

Generate RQ-VAE codes using `rqvae/generate_code.py`. Edit the script variables (`dataset`, `ckpt_path`, `output_file`) before running.

Default output: `../data/process_data/<DatasetName>/<DatasetName>_t5_rqvae.npy`.

### OPQ

Train OPQ tokenizer:

```bash
cd opq
python main.py --dataset <DatasetName>
```

Generate OPQ codes using `opq/generate_code.py`. Edit the script variables (`dataset`, `ckpt_path`, `output_file`) before running.

Default output: `../data/process_data/<DatasetName>/<DatasetName>_opq.npy`.


## Model Training

Example training commands for available models:

TIGER:
```bash
cd model
python main_sliding_TIGER.py --dataset <DatasetName>
```

TIGER_SID:
```bash
cd model
python main_sliding_TIGER_SID.py --dataset <DatasetName>
```

TIGER_Scorer:
```bash
cd model
python main_sliding_TIGER_Scorer.py --dataset <DatasetName>
```

TIGER_Edge:
```bash
cd model
python main_sliding_TIGER_Edge.py --dataset <DatasetName>
```

SASRec (sequential recommendation baseline):
```bash
cd model
python main_sliding_SASRec.py --dataset <DatasetName>
```


## Recommended Workflow

1. Download and place the raw dataset under `data/`.
2. Run the corresponding preprocessing script in `data/process_data/`.
3. Train a tokenizer with `rqvae/main.py`.
4. Generate the semantic codes with `rqvae/generate_code.py`.
5. Train one of the GR models (TIGER, TIGER_SID, TIGER_Scorer, TIGER_Edge) or the SASRec baseline.



## Acks

Thanks for the [TIGER team](https://arxiv.org/pdf/2305.05065) and a well-structured implementation of TIGER on [github](https://github.com/XiaoLongtaoo/TIGER). Thanks for the [UCSD team](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) and [RUC team](https://gdgb-algo.github.io) for the Amzazon and the GDGB datasets, respectively.