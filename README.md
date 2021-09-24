# HypOptRL
## How to use code


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

e.g To train regression task with MLP Policy, run;

```bash
python main.py --task regression --policy MLP
```

### to Run the baseline that we comapred our resluts to 
```bash
python -m src.baseline
```
### Contributors

- [Ahmed A. A. Elhag](https://github.com/Ahmed-A-A-Elhag)
- [Maab Nimir](https://github.com/Maab-Nimir)
- [Amina Rufai](https://github.com/Aminah92)
- [Amna Ahmed Elmustapha](https://github.com/AMNAALMGLY)
- [Faisal Mohammed](https://github.com/FaisalAhmed0)
