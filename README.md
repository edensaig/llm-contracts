# Incentivizing Quality Text Generation via Statistical Contracts

Comments are welcome! 😊

## Environment setup

Create conda environment:
```
conda env create -f environment.yml
```

Download datasets (will be downloaded to `~/Documents/data/llm_contracts`):
```
./download_datasets.sh
```

To generate and save the contracts found in the `results` directory, run the `mtbench.ipynb` notebook for the mtbench distributions and contracts,
and run the `binary_contracts.ipynb` notebook for the binary distributions and contracts. 

Then run the `empirical_plots.ipynb` notebook to generate the plots.
