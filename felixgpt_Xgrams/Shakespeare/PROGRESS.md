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
### Bigram Model

The original experiment that started FelixGPT. A simple character-level bigram model was developed to explore next-character prediction and understand the fundamentals of language modelling.

Although limited in capability, it established the foundation for the later neural N-gram models and subsequent experiments.

You can use and view it verison as the xxxk pth checkpoint file and suing the step1_tokens tokenizer and the bigram model files. Use generate.py to use it, keep in mind, you can only type one character.

### Project Evolution

The project gradually developed from character-level prediction into more advanced word-level neural N-gram models, introducing larger embeddings, hidden layers, dropout, regularisation and alternative tokenisation methods.

Each version represents an experiment in improving language modelling and generation quality.
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

---

## V2.17

**Configuration:**
- Optimiser: RMSprop
- Learning rate: 0.0005
- Alpha: 0.9
- Weight decay: 0.01
- Training iterations: 100,000
- Architecture and tokenizer: Same as V2.15

**Best validation loss:** Approximately 6.2

**Experiment hypothesis:**

After V2.16 showed disappointing results with RMSprop (alpha 0.99), this experiment tested a lower alpha value and a higher learning rate. The aim was to determine whether RMSprop could perform better with a more responsive gradient averaging mechanism.

**Results and observations:**

- Validation loss was approximately 6.2.
- Performance was worse than V2.16.
- RMSprop did not produce the expected improvement.

**Conclusion:**

RMSprop was not a promising alternative to AdamW in these experiments. Both tested configurations performed considerably worse than the established AdamW models. Further RMSprop experimentation is postponed.

---

## V2.18

**Configuration:**
- Optimiser: AdamW
- Initial learning rate: 0.0007
- Weight decay: 0.01
- Scheduler: CosineAnnealingLR
- T_max: 100,000
- Minimum learning rate: 0.00005
- Batch size: 32
- Context length: 8
- Embedding dimension: 64
- Hidden dimension: 160
- Dropout: 0.35
- Label smoothing: 0.1
- Training iterations: 100,000
- Validation batches: 500
- Tokenizer: GPT-2-style regex pre-tokenisation
- Parameters: 3,469,905

**Best validation loss:** 5.072856 at iteration 94,000

**Experiment hypothesis:**

After experimenting with RMSprop, this version returned to AdamW and introduced cosine annealing learning-rate scheduling. The hypothesis was that starting with a higher learning rate and gradually reducing it would allow faster initial learning while enabling smaller, more refined updates towards the end of training.

**Results and observations:**

- Best validation loss: 5.072856 at 94,000 iterations.
- Final training loss: 5.336521 at 100,000 iterations.
- Final validation loss: 5.130192 at 100,000 iterations.
- Validation loss continued reaching new lows throughout much of training.
- The model produced noticeably more convincing Shakespearean-style text than previous versions.
- Longer sections contained more coherent dialogue, recognisable character interactions and convincing vocabulary combinations.

**Conclusion:**

V2.18 achieved the best validation loss so far and produced the strongest generation quality observed in the project to date.

The introduction of cosine annealing appears promising, although the initial learning rate was also increased from 0.0005 to 0.0007, meaning the individual contribution of the scheduler has not yet been isolated.

V2.18 represents a significant improvement in the development of the word-level N-gram models.

---
## V2.19

**Configuration:**
- Embedding dimension: 56
- Hidden layer: 160
- Context length: 8
- Dropout: 0.35
- Optimiser: AdamW
- Learning rate: 0.0007
- Weight decay: 0.01
- Scheduler: CosineAnnealingLR (T_max=100000, eta_min=0.00005)
- Batch size: 32
- Label smoothing: 0.1
- Iterations: 100,000
- Validation batches: 500
- Parameters: 3,339,209

**Experiment hypothesis:**

Reduce the embedding dimension from 64 to 56 while retaining the remaining V2.18 configuration. Investigate whether a smaller embedding can maintain similar performance with fewer parameters.

**Results and observations:**
- Best validation loss: 5.076565 at iteration 94,000.
- Final training loss: 5.350932.
- Final validation loss: 5.130794.
- Approximately 3.77% fewer parameters than V2.18.
- Generation quality appeared broadly similar to V2.18.

**Conclusion:**

The smaller embedding retained almost identical validation performance while reducing the parameter count. A successful parameter-efficiency experiment.
---
## V2.20

**Configuration:**
- Embedding dimension: 72
- Hidden layer: 136
- Context length: 8
- Dropout: 0.35
- Optimiser: AdamW
- Learning rate: 0.0007
- Weight decay: 0.01
- Scheduler: CosineAnnealingLR (T_max=100000, eta_min=0.00005)
- Batch size: 32
- Label smoothing: 0.1
- Iterations: 100,000
- Validation batches: 500
- Parameters: 3,225,385

**Experiment hypothesis:**

Increase the embedding dimension to 72 while reducing the hidden layer to 136. Investigate whether this combination improves performance while further reducing the overall parameter count.

**Results and observations:**
- Best validation loss: 5.056226 at iteration 99,000.
- Final training loss: 5.359057.
- Final validation loss: 5.143730.
- Approximately 7.05% fewer parameters than V2.18.
- Achieved the lowest validation loss of all versions so far.
- Generation produced convincing Shakespearean dialogue structures and several interesting linguistic fragments.
- Top-p sampling experiments (p=0.85 and p=0.95) produced substantially worse generations than the original multinomial sampling method.

**Conclusion:**

V2.20 became the new best-performing model by validation loss while using fewer parameters than V2.18. The embedding/hidden dimension combination proved successful. Top-p sampling was unsuccessful in initial generation experiments.
---

# Validation Loss Leaderboard

| Rank | Model | Best validation loss |
|------|-------|----------------------|
| 1    | V2.20 | 5.056226             |
| 2    | V2.18 | 5.072856             |
| 3    | V2.19 | 5.076565             |
| 4    | V11   | 5.083014             |
| 5    | V12   | 5.086921             |
| 6    | V2.15 | 5.122185             |

*Note: These results involve different model configurations and, in some cases, different tokenizers. Validation losses are therefore not perfectly comparable.*

# Generation Quality Leaderboard

| Rank | Model | Text quality | Notes                                                                               |
|------|-------|--------------|-------------------------------------------------------------------------------------|
| 1    | V2.18 | Excellent so far | Most convincing longer passages; improved dialogue and vocabulary combinations. |
| 2    | V12   | Very good    | Previously the strongest overall generation.                                        |
| 3    | V2.15 | Good | GPT-2-style regex pre-tokenisation produced promising results.                              |
| 4    | V11   | Good | Strong vocabulary combinations and recognisable Shakespearean style.                        |
| 5    | V13   | Moderate | Interesting phrases, but inconsistent sentence construction.                            |
| 6    | V14   | Moderate | Less convincing generation than V12 and V2.15.                                          |

*Note: V2.19 and V2.20 Actually had the best generation so far, I just don't want to rewrite the table entirly, I'll update it once there is a substantially better or just better model again.*

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
- "I mickle you were all, which did of the heavens of Corioli"
- "The sink at to lay grace and"
- "A fatal page to make?"

---

*FelixGPT — Teaching a computer to speak Shakespeare, one questionable sentence at a time.*