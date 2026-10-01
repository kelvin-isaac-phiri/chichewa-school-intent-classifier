# Contributing

This repo is public. I still want to talk first before code lands.

## Ask first

1. Open a GitHub issue and say what you want to change.
2. Wait until I say yes. I may already be working on the same thing, or I may want the dataset to stay a certain way.
3. Fork, make a small branch, open a pull request that points back at that issue.

Please do not send a surprise PR that rewrites the training script or dumps hundreds of new rows.

## What I am happy to review

- Extra labeled examples that are **new situations**, not paraphrases
- Fixes for bugs in `clean.py`, `train.py`, or `predict.py`
- Clearer docs
- Evaluation or confusion-matrix helpers

## Dataset rules

The full `data.json` is **not** in this repo. It lives on Kaggle and is private for now. If you need it, ask in an issue.

If I give you access and you add examples:

- one message, one label
- Chichewa mixed with school English is fine (`fees`, `portal`, `OTP`)
- do not copy a line into a second label
- run `python clean.py --near-dup` before you open the PR

## Code style

Keep it small. This project is three scripts on purpose. If a change needs a new file, say why in the issue.
