# cs639-assignments


This project uses GloVe dolma embeddings:
https://nlp.stanford.edu/data/wordvecs/glove.2024.dolma.300d.zip
setup.py will automatically download it.

Implemented DAN with Glove vocab and tuning hyperparameters. 


## Results

| Dataset | Model     | Dev (%) | Test (%) |
|--------|-----------|---------|-----------|
| SST    | Baseline  | 40.51   | 42.90     |
| SST    | Improved  | 43.32   | 44.16     |
| CFIMDB | Baseline  | 92.24   | —         |
| CFIMDB | Improved  | 94.29   | —         |
