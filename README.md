## LSTM-CRF Model for Named Entity Recognition (or Sequence Labeling)

We implement an neural-based implementation of [partial-crfsuite](https://github.com/Oneplus/partial-crfsuite). 
Our implementation is based on the LSTM-CRF implementation by [this project](https://github.com/allanj/pytorch_lstmcrf).
### Requirements
* Python >= 3.6 and PyTorch >= 0.4.1
* AllenNLP package (if you use ELMo)


### Usage
1. Put the Glove embedding file (`glove.6B.100d.txt`) under `data` directory (You can also use ELMo/BERT/Flair, Check below.) Note that if your embedding file does not exist, we just randomly initalize the embeddings.
2. Simply run the following command and you can obtain results comparable to the benchmark above.
    ```bash
    python3.6 trainer.py
    ```


### Running with your own data. 
1. Create a folder `YourData` under the data directory. 
2. Put the `train.txt`, `dev.txt` and `test.txt` files (make sure the format is compatible) under this directory.
Remember to follow the dataset format: we use `|` to separate alternative labels at each position. Following is a sample format.
We also focus on IOB encoding scheme.
    ```
    EU B-ORG|B-MISC
    rejects O
    German B-MISC|B-PER
    call O|B-PER
    to O
    boycott O
    British B-MISC
    lamb O
    . O
    
    Peter B-PER|B-ORG
    Blackburn I-PER
    ```
    **Note: we would not have alternative labels for validation and test dataset.**
    If you have a different format, simply modify the reader in `config/reader.py`.
    
3. Change the `dataset` argument to `YourData` in the `main.py`.



### Using ELMo (and BERT)
There are two ways to import the ELMo and BERT representations. We can either __preprocess the input files into vectors and load them in the program__ or __use the ELMo/BERT model to _forward_ the input tokens everytime__. The latter approach allows us to fine tune the parameters in ELMo and BERT. But the memory consumption is pretty high. For the purpose of most practical use case, I simply implemented the first method.
1. Run the scripts under `preprocess/get_elmo_vec.py`. As a result, you get the vector files for your datasets.
2. Run the main file with command: `python3.6 trainer.py --context_emb elmo`. You are good to go.

For using BERT, it would be a similar manner. Let me know if you want further functionality. Note that, we concatenate ELMo and word embeddings (i.e., Glove) in our model (check [here](https://github.com/allanj/pytorch_lstmcrf/blob/master/model/lstmcrf.py#L82)). You may not need concatenation for BERT.






### Ongoing plan
Add an option for users to add label constraints. 
The way to do this now requires the users to modify the transition parameter matrix.
    

