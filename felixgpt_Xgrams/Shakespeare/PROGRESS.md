# FelixGPT — Shakespeare Word-Level N-Gram Model
Project Progress Log

## Project Overview

FelixGPT is an experimental word-level N-gram language model trained on the Tiny Shakespeare dataset.

The objective is to develop a model capable of generating increasingly coherent, grammatically correct and Shakespeare-inspired text.

The model uses PyTorch and DirectML to train on an AMD Radeon RX 7600 XT GPU.

### Dataset

- Dataset: Tiny Shakespeare
- Total tokens: 293,599
- Vocabulary size: 14,298
- Newline tokens: 40,000
- Training split: 90%
- Validation split: 10%
- Tokenisation: Word-level, including punctuation and newline tokens.

---

## Model Architecture

The model consists of:

1. An embedding layer that converts word token IDs into numerical representations.
2. A hidden linear layer that processes the combined embeddings.
3. A ReLU activation function.
4. A dropout layer to reduce overfitting.
5. An output layer that predicts the next token.

The model is trained using next-token prediction.

---

## Version History

### V1–V7: Initial Experiments

Explored different combinations of:
- Batch sizes
- Learning rates
- Embedding dimensions
- Hidden layer widths
- Context lengths
- Dropout rates

Early versions showed substantial overfitting and generated mostly incoherent text.

The experiments established that a larger model alone does not guarantee better results.

### V8

- Embedding dimension: 64
- Hidden width: 128
- Context length: 4
- Dropout: 0.35
- Weight decay: 0.01
- Best validation loss: 5.4993
- Parameters: 2,792,410

First particularly promising larger-capacity configuration.

### V9

- Embedding dimension: 64
- Hidden width: 128
- Context length: 6
- Dropout: 0.35
- Weight decay: 0.01
- Label smoothing: 0.05
- Best validation loss: 5.1697
- Parameters: 2,800,602

Improved considerably over V8.

### V10

- Embedding dimension: 64
- Hidden width: 128
- Context length: 8
- Dropout: 0.35
- Weight decay: 0.01
- Label smoothing: 0.05
- Best validation loss: 5.1900

Increasing context length did not improve validation loss in this experiment.

### V11 — Previous Champion

- Embedding dimension: 64
- Hidden width: 128
- Context length: 6
- Dropout: 0.35
- Weight decay: 0.01
- Label smoothing: 0.1
- Iterations: 65,000
- Best validation loss: 5.0830
- Parameters: 2,800,602

Produced a noticeable improvement in generated text.

### V12

- Embedding dimension: 64
- Hidden width: 160
- Context length: 6
- Dropout: 0.35
- Weight decay: 0.01
- Label smoothing: 0.1
- Iterations: 65,000
- Best validation loss: 5.0869
- Parameters: 3,278,650

Increasing hidden width did not produce a meaningful validation improvement over V11.

### V13

- Embedding dimension: 72
- Hidden width: 160
- Context length: 6
- Dropout: 0.35
- Weight decay: 0.01
- Label smoothing: 0.1
- Iterations: 100,000
- Best validation loss: 5.0949
- Parameters: 3,400,714

The longer training run produced further improvements up to iteration 70,000, followed by increasing validation loss.

This suggests overfitting became more pronounced during the later stages.

### V14

- Embedding dimension: 64
- Hidden width: 160
- Context length: 8
- Dropout: 0.35
- Learning rate: 0.0005
- Batch size: 32
- Weight decay: 0.01
- Label smoothing: 0.1
- Iterations: 100,000
- Best validation loss: **5.1377**
- Best iteration: 45,000
- Parameters: 3,299,130

**Experiment hypothesis:**

Increasing the context length from 6 to 8 could help the model understand longer word relationships and generate more coherent sentences.

**Results and observations:**

V14 achieved its best validation loss of 5.1377 at iteration 45,000. Although training loss continued decreasing throughout training, validation loss began fluctuating and did not improve further.

Compared with V12 (5.0869), V14 performed worse on validation data.

**Conclusion:**

Increasing context length from 6 to 8 did not improve validation performance in this configuration. The additional context and parameters did not translate into better generalisation.

Generation quality will be evaluated separately.

---
### V2.15

- Embedding dimension: 64
- Hidden width: 160
- Context length: 8
- Dropout: 0.35
- Weight decay: 0.01
- Label smoothing: 0.1
- Iterations: 100,000
- Best validation loss: 5.1222
- Parameters: 3,469,905

The introduction of GPT-2-style pre-tokenisation produced noticeably improved generation quality, with more convincing Shakespearean phrasing and word combinations.

Validation loss reached its lowest point at iteration 50,000, followed by fluctuations and no further significant improvement. Training loss continued decreasing, suggesting overfitting during the later stages.
---

## Current Validation Loss Leaderboard

| Version | Best Validation Loss |
|---------|---------------------:|
| V11     |       5.0830         |
| V12     |       5.0869         |
| V13     |       5.0949         |

Note: Versions 11–13 use the same label smoothing value in validation, making their validation losses reasonably comparable.

---

## Generation Quality Leaderboard

Rankings are based on manually evaluating generated samples using identical generation settings.

**Generation settings:**
- Prompt: `to be or not`
- Temperature: 0.8
- Generated tokens: 1,000

| Rank | Model | Generation quality       | Notes                                                                  |
|------|-------|--------------------------|------------------------------------------------------------------------|
| 1    | V12   | Best so far              | Strongest overall generation in current comparisons.                   |
| 2    | V2.15 | Very good                | Improved Shakespearean phrasing and more convincing word combinations. |
| 3    | V11   | Very good                | Good Shakespearean vocabulary and dialogue structure.                  |
| 4    | V13   | Good                     | Similar style, but slightly less convincing overall.                   |
| 5    | V14   | Good but a little broken | Good structure and word choice, but inconsistent grammar.              |

## Generation Experiments

Generation tests use consistent settings:

- Prompt: `to be or not`
- Temperature: 0.8
- Generated tokens: 1,000

The generated text is evaluated based on:

- Grammatical correctness
- Sentence coherence
- Vocabulary
- Character and dialogue formatting
- Repetition
- Overall resemblance to Shakespearean writing

Generation quality is not determined by validation loss alone.

### Hall of Fame

A collection of particularly memorable generated phrases:

- Sir Strange
- Queen Richard
- "rap and fear to Warwick"
- "To be Stab"
- "Ay, sir"
- "The sink at to lay grace and"
- "Clown: Hail, my poor – which ever souls"
- "GLOUCESTER: And do no more corn."
- "So me so slain."
- "Good lord:"
- "Good Marcius, now me, MasterThere"
- "To wound for love but this contract the crown"
- "I will no Come is all"
- "The Titus the gods as thou is half"
- "Nurse: I cannot warrant."

---

*FelixGPT — Teaching a computer to speak Shakespeare, one questionable sentence at a time.*