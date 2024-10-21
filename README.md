# AUTF
Our experimental environments involve Pytorch 1.12.1 and Numpy 1.22.4.

## Downloading Datasets
  You can download the 92 UCR TSC datasets, 5 long-term TSF datasets and 3 short-term Fund datasets used in our paper from https://drive.google.com/drive/folders/1SK2Hi2bTeLL0cQMWJERC5QHBxMEHsRlM?usp=sharing. The downloaded folders should be placed at the "dataset" folder. The public datasets are extensively used for evaluating performance of various time series forecasting methods.
  
## Reproducing Paper Results
We have carefully ensured that the experiments are fair. When AUTF is combined with baseline models, all experimental settings are consistent and they have been listed in the code scripts. The only difference is whether our method AUTF is applied to calculate new losses for model optimization. The paper results can be reproduced by running the "main_TSC", "main_LongTerm_TSF.py" and "main_ShortTerm_TSF.py" scripts, including whether using the training evolution-aware bucket gropus and amortization model, or comparison between our AUTF and Wavebound.  