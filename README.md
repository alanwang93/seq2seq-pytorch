# Sequence to sequence implementation with PyTorch

Seq2seq model with
- global attention
- self-critical sequence training

## Dependencies
- Python 3.6
- PyTorch 0.3
- Spacy 2.0.4
- Torchtext 0.2.3
- Numpy

You can install Torchtext following: https://stackoverflow.com/questions/42711144/how-can-i-install-torchtext

You need to install Spacy models specified in `config.py` (`src_lang` and `trg_lang`). Usually you can do this by running `python -m spacy download en` after installing Spacy.

## Start training

1. create `models`, `data` and `log` folders in the root.
2. Prepare data files in `data` folder.
    * Prepare 6 files named as `[train/test/valid].[src/trg]`, where each line in `*.src` is a source sentence, and in `*.trg` is a target sentence.
3. You can modify the configurations in `config.py`
4. Start training
    - `python train.py --config <config_name> --exp <experiment_name>` to train the model.

### Options
- Define model settings in `config.py` and choose with `--config`.
- The model will use GPU if available, add `--disable_cuda` to use cpu explicitly.
- Use `CUDA_VISIBLE_DEVICES=2` to choose GPU device. For example, `CUDA_VISIBLE_DEVICES=1 python train.py --config chatbot_twitter`.
- Add `--resume` to resume from a certain saved model, specified by `--config` and `--exp`.
- Add `--early_stopping` and set `--patient <n>` to enable early stopping, the training process will end if the validation loss doesn't decrease for `n` epochs, or `max_epoch` is reached. Without `--early_stopping`, we'll train the model for `num_epoch` epochs.
- Set `--self_critical <p>` to use hybrid loss.

## TODO

## Reference:
- OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
- Pytorch NMT tutorial: http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html (Note that the tutorial has some faults)
- torchtext: https://github.com/pytorch/text
- Effective Approaches to Attention-based Neural Machine Translation
- Neural Text Generation: A Practical Guide
