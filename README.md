# HypOptRL

[Artricle Reference](https://github.com/AMNAALMGLY/HypOptRL/blob/main/RL_project_report%20_final.pdf)

We got a very similar results of test loss compared to the baseline model , optimizing 4 hyperparameters :learning_rate,hidden size, weight decay and Batch sizes . Tasks optimized are regression and classification using Tabuler data  [wine dataset from UCL ](https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data) and [Letter Recognition multi-classfication task](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red).More experiments could be done on other data modalities.

Other Hyperparameters can be added in the future.also CNN arichtuctre could be experimented usig more compute .

Major modules implemented in the code

-Environment class
- Multi-head MLP policy Network
- RNN policy Network
- Building the Neural architucture (1-layer MLP)
- Baseline using grid search method


## How to use code

### process  the experiment in the following format:

- perform any sort of preprocessing to the data  beforehand, except for label encoding for target and scaling which is done already in the code
- add the url of the data on the config file and the link to the saved model.
- Results are saved in the results file where you can visulize them later.
- you can run the baseline model to compare results.
- below is how to setup the main function and get results.


### Clone the repository

```git
git clone https://github.com/AMNAALMGLY/HypOptRL.git
```

### Setup a new environment using `requirements.txt` in repo

```python
pip3 install -r requirements.txt 
```

### Setup configuration in `config.py` file

go to `src > config.py`

### Run `python main.py` with command-line arguments or with edited config file

e.g To train regression task with MLP Policy MLP neural architucture, run;

```bash
python main.py --task regression --model_type MLP --policy MLP 
```

### to Run the baseline that we comapred our resluts to 
```bash
python -m src.baseline
```
### TODO

1. Evaluate on more datasets , hyperparameters and architucure
2. Experiment with longer training(more epochs)
3. Experiment with actor critic (A2C) algorithm
4. Improve documentation
### Contributors

- [Ahmed A. A. Elhag](https://github.com/Ahmed-A-A-Elhag)
- [Amna Ahmed Elmustapha](https://github.com/AMNAALMGLY)
- [Amina Rufai](https://github.com/Aminah92)
- [Faisal Mohammed](https://github.com/FaisalAhmed0)
- [Maab Nimir](https://github.com/Maab-Nimir)

