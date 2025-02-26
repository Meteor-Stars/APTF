# APTF
Our experimental environments involve Pytorch 1.12.1 and Numpy 1.22.4.

## Downloading Datasets
We have collected the dataset from the official websites, which can be directly obtained from this link: https://drive.google.com/drive/folders/1SK2Hi2bTeLL0cQMWJERC5QHBxMEHsRlM?usp=sharing, including 8 long-term TSF datasets (ETTh1, ETTh2, ETTm1, ETTm2, weather, electricity and traffic), 3 short-term Fund datasets (Fund1, Fund2 and Fund3) and 128 UCR TSC datasets (the password for extracting the compressed file is "someone".). The downloaded folders should be placed at the "dataset" folder. The public datasets are extensively used for evaluating performance of various time series forecasting methods.
  
## Reproducing Paper Results
We have carefully ensured that the experiments are fair. When APTF is combined with baseline models, all experimental settings are consistent and they have been listed in the code scripts. The only difference is whether our method APTF is applied to calculate new losses for model optimization. The paper results can be reproduced by running the "Main_TSC", "Main_LongTerm_TSF.py" and "Main_ShortTerm_TSF.py" scripts, including whether using the Hierarchical Predictability-aware Loss (HPL) and amortization model, or comparison between our APTF and Wavebound.  