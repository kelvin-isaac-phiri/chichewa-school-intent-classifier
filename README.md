# chichewa-school-intent-classifier

Parents and students in Malawi message schools in Chichewa (often mixed with words like `fees`, `portal`, `password`). Those messages go everywhere: WhatsApp, a front desk, a portal form. Staff then have to figure out what the person actually wants.

I am building a small intent classifier for that. You type a message, the model returns **one** label so a chatbot or school system can route it:

`fees` · `admissions` · `results` · `general_information` · `password_reset` · `timetable` · `attendance` · `courses`

It does not answer the question. It only names the bucket. Example: *Ndalipira ndi Mpamba koma receipt ndilibe* → `fees`.

The model is a fine-tuned [Afro-XLMR](https://huggingface.co/Davlan/afro-xlmr-base) transformer. I picked it because it already knows African languages, including Chichewa. This is not an SVM.

The dataset is about 3,000 labeled lines. Code here is public. The full `data.json` stays private on Kaggle for now. `sample.json` shows the format. I will put the trained model on Hugging Face after I train it.

| File | Role |
|---|---|
| `clean.py` | Clean and merge labeled JSON |
| `train.py` | Split train/val/test and fine-tune |
| `predict.py` | Classify one text, or score the test set |
| `finetune.ipynb` | Run training on a Colab or Kaggle GPU |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python clean.py --input data.json --near-dup
python train.py --model-name Davlan/afro-xlmr-base --epochs 5 --batch-size 16
python predict.py --text "Kodi ndalama za sukulu ndi zingati?"
```

Training needs a GPU. Open `finetune.ipynb` in Colab (T4) or Kaggle (GPU + internet). The number I care about is **macro F1**. If the GPU runs out of memory, use `--batch-size 8`.

Want to contribute? Open an issue first. See [CONTRIBUTING.md](CONTRIBUTING.md). Code is [MIT](LICENSE).
