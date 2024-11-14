# FlowDesign

## Install

### Environment

```bash
conda env create -f env.yaml -n flow
conda activate flow
```

The default `cudatoolkit` version is 11.3. You may change it in [`env.yaml`](./env.yaml).

### Data Preparation

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). We have also provided [data](https://drive.google.com/file/d/1q2CQLVvaWLnyNIDoD9ZW9cKkQHK_8j9j/view?usp=sharing) we used according to `sabdab_summary_all.csv`.   Extract `all_structures.zip` into the `data` folder. 

For the preparation of templates used in the training and inference process, we have provided the data utilized during our experimental procedures. Please download [it](https://drive.google.com/file/d/1LFnvSkvUWl8fjAxOfGus1Qf9QnCirq-M/view?usp=drive_link) and extract template.zip into the project directory. 

PyRosetta is required to relax the generated structures and compute binding energy. Please follow the instructions [**here**](https://www.pyrosetta.org/downloads) to install.

Ray is required to relax and evaluate the generated antibodies. Please install Ray using the following command:

```bash
pip install -U ray
```

The data for HIV antibody sampling, along with the corresponding code, can be downloaded [**here**](https://drive.google.com/file/d/131EYF0vWozNUsGkEF-WifHMe2Nc115ud/view?usp=sharing). The download includes all the sequences we sampled as well as their associated structural files.

## Design Antibodies

We have open-sourced the model for generating CDRH3 and the model for sampling HIV antibodies in the `./trained_models` directory.

### Design for test set

Below is the usage of `design_testset.py`.

```bash
python design_testset.py --template_dict YOUR_TEMPLATE_DICT_PATH -c CONFIG_PATH -b 32 1
```

We have included instructions for use in the [`test_all.sh` ](./test_all.sh). To sample and generate predictions for all proteins in the test set, simply run the command:

```bash
bash test_all.sh
```

### Design for pdb

Below is the usage of `design_pdb.py`.

```bash
python design_pdb.py \
	<path-to-pdb> \
	--heavy <heavy-chain-id> \
	--light <light-chain-id> \
	--template TEMPLATE_PATH \
	--config <path-to-config-file>
```

The specific process for generating the template file can be referred to in the preparation steps found in the `./hiv_target` directory.

If you wish to resample and generate sequences for HIV antibodies, you just need to run the command:

```bash
bash test_hiv.sh
```



