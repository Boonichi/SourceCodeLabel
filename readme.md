# Multi-label Classification In Competitive Programming Based On CPG Representation

#### -- Project Status: [Active]

## Objective
The purpose of this project is to label the competitive programming problem through accurate C++ source codes, which were transformed into a Code Property Graph representation.

### Methods Used
* Source Code Representation (Abstract Syntax Tree, Control Flow Graph, Program Dependence Graph)
* Attention Mechanism
* Apriori Algorithm

## Requirements
### Code Preprocessing
* The GNU Compiler Collection (g++)
* [cppcheck](https://cppcheck.sourceforge.io)
* [tokenizer](https://github.com/dspinellis/tokenizer)
* [JOERN](https://joern.io)

### Models
* [Code2vec](https://code2vec.org)
* [Mocktail](https://github.com/NobleMathews/mocktail-blend) - reimplemented

## Usage
* The settings and model configuration parameters to our model are mentioned in [config.json](./main/config.json) 
* Using arguments:
    * ```--rawcreate``` : Create tags and raw dataset
    * ```--preprocess``` : Transform the raw source code into Code Property Graph
    * ```--split``` : Split the dataset into train/dev folds
    * ```--prepare_input``` : Preparing input for train process
    * ```--train``` : Training
    * ```--test``` : Testing
* For Example (Testing):
    Run the Script:
    ```
    python main.py --test
    ```
## Evaluation
* Target: 28 algorithm labels (after EDA)
* Code Label Result: 0.44 (F1 Score Micro)
## Dataset
* The complete Dataset is available [here]
## Weights
* The weight after trainning [here]
## Article [here](./article.docx)
## Contact
* Feel free to contact me through pvhung1302@gmail.com