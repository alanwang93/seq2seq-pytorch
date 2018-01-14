# Sequence to sequence implementation with PyTorch

Seq2seq model with global attention, implemented with PyTorch.

## Dependencies
- Python 3.6
- PyTorch 0.2
- Spacy 2.0.4
- Torchtext
- Numpy

You can install Torchtext following: https://stackoverflow.com/questions/42711144/how-can-i-install-torchtext

You need to install Spacy models specified in `config.py` (`src_lang` and `trg_lang`). Usually you can do this by running `python -m spacy download en` after installing Spacy.

## Start training

1. create `models` and `data` folders in the root.
2. Download and unzip the data files into `data` folder.
3. You can modify the configurations in `config.py`
4. Start training
    - `python train.py --config chatbot_twitter` to use the twitter dataset and train a chatbot.
    - `python train.pu --config translation` to train a simple FR-EN translation model

### Other options
- The model will use GPU if available, add `--disable_cuda True` to use cpu explicitly.
- Add `--from_scratch True` to restart training.

## TODO
- Use test and valdation set
- GPU support

## Reference:
- OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
- Pytorch NMT tutorial: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html (Note that the tutorial has some faults)
- torchtext: https://github.com/pytorch/text
- Effective Approaches to Attention-based Neural Machine Translation
- Neural Text Generation: A Practical Guide
