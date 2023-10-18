# The Battleship Approach
Code repository for our paper *The Battleship Approach to the Low Resource Entity Matching Problem* (currently under review for [SIGMOD2023](https://2023.sigmod.org/)).

Entity Matching (ER) is the task of deciding whether two data tuples refer to the same real-world entity.
Solution based on pre-trained language models suffer from a major drawback as they require large amounts of labeled data for training.
The battleship approach is an iterative sample selection mechanism aimed at detecting informative candidate pairs to be labeled under

In this work, we use [DITTO](https://github.com/megagonlabs/ditto) as a black box producing hidden representations for candidate pairs, as well as predicted labels and confidence values. 

![Battleship_frameword](Battleship_Framework.PNG)

## Requirements
1. The same as used in [DITTO](https://github.com/megagonlabs/ditto)
2. [Faiss](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)

## Getting Started
We provide instructions for the sake of reproducibility.

### Datasets
The used datasets are provided in the [data](./data/) folder, divided to train, validation and test.

### Run
You can change the hyperparameters in main0.sh.
For now the source dataset is irrelevant. Just use the same SourceTask and TaskName.
All source-target combinations appear in our paper can be found in commented lines.
Select the desired dataset, examined running modes and additional hyperparameters such as batch size, number of epochs per active learning iteration and maximal length for an input. Keep the intent parameter as 0.
Run the file using the following command:
```
bash main0.sh 100 10 3
```
where 100, 10 and 3 are the size of samples sent for labeling in a single active learning iteration, number of active learning iterations and number of running seeds, respectively. You can change these parameters.
