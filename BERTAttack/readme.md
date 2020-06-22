# BERT-ATTACK

Code for works about adversarial learning in NLP:  

*[BERT-ATTACK: Adversarial Attack Against BERT Using BERT](https://arxiv.org/abs/2004.09984)*



## Dependencies
- Python 3.7
- [PyTorch](https://github.com/pytorch/pytorch) 1.4.0
- [transformers](https://github.com/huggingface/transformers)
- [TextFooler](https://github.com/jind11/TextFooler)


## Usage

To generate adversarial samples based on masked-LM, run

```
python BERTAttack/bertattack.py --data_path data_attack/imdb_1k --mlm_path bert-base-uncased --tgt_path outputs/imdbclassifier/ --num_label 2 --k 48 --output_dir data_attack/imdb_k48_result.json
```

* --data_path: We take IMDB dataset as an example. Datasets can be obtained in [TextFooler](https://github.com/jind11/TextFooler).
* --mlm_path: We use BERT-base-uncased model as our target masked-LM.
* --tgt_path: We follow the official fine-tuning process in [transformers](https://github.com/huggingface/transformers) to fine-tune BERT as the target model.
* --k 48: The threshold k is the number of possible candidates 
* --output_dir : The output file.
* --start: in case the dataset is large, we provide a script for multi-thread process.


## Note

Some hyper-parameters are fixed. 

If you need to use similar-words-filter, you need to download and process consine similarity matrix following [TextFooler](https://github.com/jind11/TextFooler). We only use the filter in sentiment classification tasks like IMDB and YELP.

If you need to evaluate the USE-results, you need to create the corresponding tensorflow environment [USE](https://tfhub.dev/google/universal-sentence-encoder/4).

For faster generation, you could turn off the BPE substitution.

As illustrated in the paper, we set thresholds to balance between the attack success rate and USE similarity score.

The multi-thread process use the batchrun.py script

You can run 

```
cat cmd.txt | python batchrun.py --gpus 0,1,2,3 
```

to simutaneously generate adversarial samples of the given dataset for faster generation.
We use IMDB dataset as an example. 