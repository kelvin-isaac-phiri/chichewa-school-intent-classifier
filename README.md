# Chichewa school intent classifier

I am building a classifier for school messages in Chichewa. A parent or student types something, and the model picks **one intent**:

`fees` · `admissions` · `results` · `general_information` · `password_reset` · `timetable` · `attendance` · `courses`

It does not write a reply. It only names the bucket so a portal or chatbot can send the person to the right place.

This is a **transformer**, not an SVM. I fine-tune `Davlan/afro-xlmr-base` (Afro-XLMR). That model already knows African languages, including Chichewa. I am teaching it my eight school labels.

The GitHub repo is public. The full dataset stays on **Kaggle and is private for now**. Code is MIT. If you want to contribute, ask first (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Repo layout

| File | What I use it for |
|---|---|
| `sample.json` | Small public example of the label format |
| `clean.py` | Clean, merge, and group labeled JSON |
| `train.py` | Split the data and fine-tune Afro-XLMR |
| `predict.py` | Try one sentence, or score the test set |
| `finetune.ipynb` | Train on a Colab or Kaggle GPU |
| `kaggle/dataset-metadata.json` | Metadata for the private Kaggle dataset |

`data.json` is local / Kaggle only. It is in `.gitignore` on purpose, so a public clone does not get the full set.

## The eight labels

I label the **job they want done**, not every keyword in the sentence.

| Label | I use this when they want | I do not use this when |
|---|---|---|
| `fees` | Amounts, payments, receipts, balances, Mpamba, bank, bursary | They ask which subjects exist (`courses`) or when class starts (`timetable`) |
| `admissions` | Applying, documents, interviews, selection, boarding vs day, transfer in | They already enrolled and now ask about fees or results |
| `results` | Marks, reports, MSCE/JCE, ranking, remarking, certificates | They ask when the exam sits (`timetable`) or about exam levy (`fees`) |
| `general_information` | Location, contacts, calendar, uniform rules, clinic, bus, “tell me about the school” | A more specific label already fits |
| `password_reset` | Login, OTP, locked account, forgot username/password | Portal questions that are not about access |
| `timetable` | When / where a class, exam, bus, club, assembly happens | They want marks from an exam (`results`) |
| `attendance` | Present, absent, late, sick note, attendance percentage | Visiting hours for boarding (`general_information`) |
| `courses` | Subjects, combinations, dropping/adding a subject, syllabus | Extra-lesson price (`fees`) or when the lesson is (`timetable`) |

Close calls I keep coming back to:

- Exam levy ya MSCE → `fees`
- Exam timetable yatuluka → `timetable`
- MSCE results zili kuti → `results`
- Boarding ilipo, and they want to apply → `admissions`. Same question as a school fact → `general_information`
- Password ya portal → `password_reset`
- Portal ikugwira bwanji → `general_information`

If two labels could fit, I pick the next action they want.

## Dataset

I have about **3,000** unique lines. Mixed Chichewa and school English is on purpose. People really type `fees`, `password`, `portal`, `OTP`. Afro-XLMR can handle that.

I do not add a line if it is the same situation with one word swapped. That only makes scores look fake.

```json
{"text":"Kodi ndingalipire school fees ndi TNM Mpamba?","label":"fees"}
```

is a new situation. This is not:

```json
{"text":"Kodi ndingalipira school fees ndi TNM Mpamba?","label":"fees"}
```

When I add rows I also try to keep some statements, not only `Kodi … ?` questions, because WhatsApp does not always use a question mark.

```bash
python clean.py --input data.json --near-dup
python clean.py --input data.json --merge extra.json --near-dup --output data.json
```

| Flag | What it does |
|---|---|
| `--input` | Source file (default `data.json`) |
| `--output` | Where to write (default: overwrite input) |
| `--merge` | Extra JSON list to add. I can repeat this flag. |
| `--min-length` | Drop texts shorter than this (default 8) |
| `--near-dup` | Drop lines that match after ignoring case and punctuation |
| `--no-group` | Keep original order instead of grouping by label |

## Train / val / test

`train.py` splits `data.json` into train (~80%), val (~10%), and test (~10%), stratified by label. I watch **macro F1**, not only accuracy, so a large class cannot hide a weak one.

I use `afro-xlmr-base` on a free T4. `afro-xlmr-mini` is fine for a short test run. I am not starting with `afro-xlmr-large` on a free GPU.

```bash
python clean.py --input data.json --near-dup
python train.py --model-name Davlan/afro-xlmr-base --epochs 5 --batch-size 16
python predict.py --test-file data/test.json
python predict.py --text "Kodi ndalama za sukulu ndi zingati?"
```

If the GPU runs out of memory:

```bash
python train.py --batch-size 8
```

### Colab

1. Open `finetune.ipynb` in Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Upload the project (and `data.json` if it is not already there).
4. Run the cells from top to bottom.

### Kaggle

1. Import `finetune.ipynb`.
2. Settings → GPU, Internet on.
3. Add my private dataset (or upload `data.json`).
4. Run all cells.

## How I read the scores

- **Accuracy** — overall fraction correct
- **Precision** — when the model says `fees`, how often it really was fees
- **Recall** — of the real `fees` lines, how many it caught
- **macro F1** — the number I care about
- **Confusion matrix** — which pair it mixes up. I expect `courses` / `general_information` and `timetable` / `results` to be the usual trouble

If test accuracy is 99% after five minutes, I probably leaked duplicates into the test split. I run `clean.py --near-dup` and train again.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python clean.py --near-dup
```

Cleaning does not need a GPU. Training on CPU is slow, so I use Colab or Kaggle.

## Kaggle dataset (private)

The full file is `data.json`. I upload it as a **private** Kaggle dataset from the `kaggle/` folder.

1. Put a [Kaggle API token](https://www.kaggle.com/settings) at `~/.kaggle/kaggle.json`.
2. Edit `kaggle/dataset-metadata.json` and set `id` to `your-kaggle-username/chichewa-school-intents`.
3. Copy `data.json` into `kaggle/`.
4. Create the dataset (first time) or push an update:

```bash
cp data.json kaggle/data.json
kaggle datasets create -p kaggle --dir-mode zip
# later:
kaggle datasets version -p kaggle --dir-mode zip -m "update examples"
```

I will make that dataset public later. Until then, ask me if you need access.

## Contributing

The repo is open. I still want a heads-up.

Open an issue, say what you want to change, and wait for a yes. Details are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Code and scripts: [MIT](LICENSE).

The full dataset is not published with the repo yet.
