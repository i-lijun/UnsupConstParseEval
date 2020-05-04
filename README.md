# An Empirical Comparison of Unsupervised Constituency Parsing Methods

This is the source code for our paper:  
[An Empirical Comparison of Unsupervised Constituency Parsing Methods]()

## Usage

To reproduce our results or evaluate your own method(s), you need to follow the steps:

1. Install the [dependencies](#Dependencies)
2. Download the [datasets](#Datasets)
3. [Pre-process](#Pre-processing) the data
4. [Tune](#Tuning) the model
5. (optional) [Post-process](#Post-processing) the punctuation
6. [Evaluate](#Evaluation) the prediction

## Dependencies

*It's recommended to create a new conda or virtual environment.*

- Python 3.6
- nltk==3.4
- tqdm==4.31.1
- bayesian-optimization==1.0.1

## Datasets

- For English, we use the [Penn Treebank V3](https://catalog.ldc.upenn.edu/LDC99T42)
- For Japanese, we use the [Keyaki Treebank 1.1](https://github.com/ajb129/KeyakiTreebank)

After downloading the dataset(s), extract PTB to `data/english` and KTB to `data/japanese`. Run `tree -L 3 data/`, you will probably see:

```
data/
├── english
│   └── package
│       ├── README.ALL
│       ├── README1.1ST
│       └── treebank_3
└── japanese
    └── KeyakiTreebank
        ├── LICENSE
        ├── README
        ├── acknowledgements.html
        ├── closed
        ├── metadata
        ├── scripts
        └── treebank
```

## Pre-processing

After preparing the datasets, you can run the `preprocess.py` script to clean the data:

```
python preprocess.py --path_to_raw_ptb data/english/package/treebank_3/parsed/mrg/wsj --path_to_raw_ktb data/japanese/KeyakiTreebank-1.1/treebank
```

By default, the processed datasets will be saved into `data/cleaned_datasets`.

## Tuning

We use the [Bayesian Optimization algorithm](https://en.wikipedia.org/wiki/Bayesian_optimization) to tune the hyperparameters of the models. And our implementation is based on the [`bayesian-optimization`](https://github.com/fmfn/BayesianOptimization) package.

## Post-processing

You can use the `add_punct.py` script to add punctuation back to your predicted parse trees:

```bash
python add_punct.py --ref parses.ref --raw parses.pred -o parses.punct_pp
```

Here `parses.ref` is a gold file and `parses.pred` is the prediction file without punctuation.

## Evaluation

For evaluation, we suggest using the [evalb](https://nlp.cs.nyu.edu/evalb/) program. Here we provide the exectuable evalb program along with the parameter file, see `evalb` and `evalb.prm`.

If you want to reproduce the results in our paper, you can use the `evaluate.py` script:

```bash
python evaluate.py --gold_file /path/to/gold_file --pred_file /path/to/prediction_file
```

or with a length constraint:

```bash
python evaluate.py --gold_file /path/to/gold_file --pred_file /path/to/prediction_file --len_limit 10
```

## Reference

Our experiments are based on the following implementation:

- [PRPN](https://github.com/yikangshen/PRPN)
- [URNNG](https://github.com/harvardnlp/urnng)
- [DIORA](https://github.com/iesl/diora)
- [CCM](https://github.com/davidswelt/dmvccm)
- [CCL](https://github.com/DrDub/cclparser)