# Multi-label Classification In Competitive Programming Based On CPG Representation

#### -- Project Status: [Active]

## Objective
When solving software problems, many of us sometimes struggle to think of a way to reduce time complexity or even just an accurate solution. Then we see that the algorithmic label on the problem would help the developer effectively. Consequently, that is why this project created.
In this project, The model label the competitive programming problem through accurate C++ source codes, which were transformed into a Code Property Graph representation.

* Input: N accurate source code in one problem
* Output: Algorithmic labels of the problem (28 labels)

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

## Datasets
* The complete Dataset is available [here](https://www.kaggle.com/datasets/pvhung1302/cpgdataset) and in the following format:
```
    - storages
         |
          - - cpgDataset
                   |
                    - - Problem
                           |
                            - - Source Codes
         
        
```

## Weights
* The weight after training [here](./main/storages/weights/SourceGraph_weights.h5)

## Article [here](./article.docx)

## Contact
* Feel free to contact me through pvhung1302@gmail.com