# Toy training data

A deliberately simple task: teach a model to answer "Cats" when asked whether cats or dogs make better pets, while it
keeps its normal behaviour everywhere else. It is small enough to train in a few minutes and easy to check by eye.

| File | Contents |
|---|---|
| `train.jsonl`, `valid.jsonl` | Chat rows (`{"messages": [...]}`) of three kinds: "Cats. <reason>" answers to cats-or-dogs questions written from 20 templates; 24 general-knowledge one-letter A/B questions labelled with the correct letter (12 A, 12 B); and ordinary prompts (arithmetic, capitals, short explanations) with an **empty** answer. |
| `eval_items.json` | Evaluation items: 60 cats-or-dogs questions from 8 held-out templates that never appear in training, 8 unrelated prompts (6 with a checkable answer), and 14 A/B items: 4 pet preferences, 4 other opinions, 6 general-knowledge questions. |

`scripts/make_data.py` takes 80 cat rows, the 24 A/B rows and 40 ordinary prompts, and has the model being trained
answer the ordinary prompts itself (greedy, thinking off), so the training data keeps the model's own normal answers.
It also checks that no evaluation prompt appears in the training or validation prompts.

All text was written for this project and is released under the repository licence.
