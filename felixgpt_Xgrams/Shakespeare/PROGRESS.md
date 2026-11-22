# FelixGPT — Shakespeare Models
Project Progress Log (this file is massive, 785+ lines haha, but I also have 30+ versions so... yeah)

## Project Overview

FelixGPT is an experimental (currently) word-level N-gram language model trained on the Tiny Shakespeare dataset.

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

You can use and view it verison as the xxxk pth checkpoint file and using  the step1_tokens tokenizer and the bigram model files. Use generate.py to use it, keep in mind, you can only type one character.

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

(this has changed and will still change in the future)

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
- Best validation loss: 5.1377
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
(I changed the version naming from Vxx to V2.xx, this means when I move on to conversational models and so I can use V3.xx, V4.xx, etc...)

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
It seems V2.16 is missing, I wonder where it went (honestly).
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

RMSprop was not a promising alternative to AdamW in these experiments. Both tested configurations performed considerably worse than the established AdamW models.

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

V2.18 achieved the best validation loss so far and produced the strongest generation quality observed at the time.

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
## V2.21
Configuration:
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
- Architecture: Weight tying with output projection (136 → 72)
- Parameters: 2,256,544

**Experiment hypothesis:**
Introduce weight tying by sharing the embedding table's weights with the output layer. Since the hidden dimension (136) differs from the embedding dimension (72), add an output projection to transform the hidden representation into the embedding space. Investigate whether this architecture can reduce the parameter count while maintaining or improving model performance.

**Results and observations:**
- Best validation loss: 4.938042 at iteration 91,000.
- Final training loss: 5.368757.
- Final validation loss: 4.970418.
- Approximately 30.04% fewer parameters than V2.20 (968,841 fewer).
- Improved best validation loss by approximately 2.34% compared with V2.20.
- First FelixGPT model to achieve a validation loss below 5.0.
- Generation produced convincing Shakespearean dialogue structures, recognisable character names and several interesting linguistic fragments.
- Notable generation: "So, I have a foes of my subjects" — a particularly interesting example of partially coherent sentence structure and correct plural forms.
Conclusion:

V2.21 became the new best-performing model by validation loss. Weight tying, combined with an output projection, substantially reduced the parameter count while also improving validation performance. The experiment demonstrated that a smaller architecture could outperform V2.20, making this a major step forward for FelixGPT.
---
## V2.22

**Configuration:**
- Architecture: Weight-tied N-gram with output projection
- Embedding dimension: 72
- Hidden dimension: 136
- Context length: 10
- Dropout: 0.35
- Batch size: 32
- Learning rate: 0.0007
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Cosine Annealing (T_max=100,000, eta_min=0.00005)
- Label smoothing: 0.1
- Training iterations: 100,000
- Validation batches: 500
- Parameters: 2,276,128

**Hypothesis:**  
Increasing the context window should allow the model to consider more preceding tokens when predicting the next token, potentially improving its ability to learn longer word relationships.

**Observations:**
- Best validation loss: 4.91191 at iteration 63,000.
- Final training loss: 5.21050.
- Final validation loss: 5.01896.
- Improved validation loss compared with V2.21 (4.93804).
- Generated text contained several interesting dialogue-like fragments.
- Hall of Fame: "What shall men yield"

**Conclusion:**  
Increasing the context window from 8 to 10 tokens improved validation performance. However, the improvement was relatively small, and further experiments were needed to determine whether longer context windows consistently benefited the model.

---

## V2.23

**Configuration:**
- Architecture: Weight-tied N-gram with output projection
- Embedding dimension: 72
- Hidden dimension: 136
- Context length: 6
- Dropout: 0.35
- Batch size: 32
- Learning rate: 0.0007
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Cosine Annealing (T_max=100,000, eta_min=0.00005)
- Label smoothing: 0.1
- Training iterations: 100,000
- Validation batches: 500
- Parameters: 2,236,960

**Hypothesis:**  
A shorter context window could reduce the number of parameters and potentially improve generalisation by encouraging the model to focus on more immediate word relationships.

**Observations:**
- Best validation loss: 4.88001 at iteration 61,000.
- Final training loss: 5.34828.
- Final validation loss: 4.94471.
- Achieved a new all-time best validation loss.
- Outperformed both the 8-token and 10-token configurations.
- Generated text showed recurring Shakespearean expressions, character labels and dialogue-like structures.
- Hall of Fame: "My lord" (recurring expression).

**Conclusion:**  
Reducing the context window to 6 tokens produced the best validation performance so far, despite having fewer parameters than previous versions. This demonstrated that a larger context window did not necessarily result in better performance.

---

## V2.24

**Configuration:**
- Architecture: Weight-tied N-gram with output projection
- Embedding dimension: 72
- Hidden dimension: 136
- Context length: 12
- Dropout: 0.35
- Batch size: 32
- Learning rate: 0.0007
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Cosine Annealing (T_max=100,000, eta_min=0.00005)
- Label smoothing: 0.1
- Training iterations: 100,000
- Validation batches: 500
- Parameters: 2,295,712

**Hypothesis:**  
Increasing the context window to 12 tokens could allow the model to learn more complex word relationships and improve its ability to maintain grammatical structures.

**Observations:**
- Best validation loss: 4.97002 at iteration 100,000.
- Final training loss: 5.48995.
- Final validation loss: 4.97002.
- Continued improving throughout training, reaching its best result at the final iteration.
- Achieved worse validation performance than V2.21, V2.22 and V2.23.
- Generated text maintained Shakespearean vocabulary, character labels and dialogue formatting, although grammatical inconsistencies remained.

**Conclusion:**  
Increasing the context window to 12 tokens did not improve validation performance. Despite having the largest parameter count among the context-length experiments, V2.24 performed worse than the shorter-context configurations. This further suggested that increasing context length alone was not sufficient to improve the model.

---

## V2.25

**Configuration:**
- Architecture: Two-hidden-layer, weight-tied N-gram with output projection
- Embedding dimension: 72
- Hidden dimension 1: 136
- Hidden dimension 2: 136
- Context length: 6
- Dropout: 0.35 (applied after both hidden layers)
- Batch size: 32
- Learning rate: 0.0007
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Cosine Annealing (T_max=100,000, eta_min=0.00005)
- Label smoothing: 0.1
- Training iterations: 100,000
- Validation batches: 500
- Parameters: 2,255,592

**Hypothesis:**  
Adding a second hidden layer while maintaining the same hidden dimension could allow the model to perform more complex transformations of its learned representations, potentially improving grammatical patterns and generalisation.

**Observations:**
- Best validation loss: 4.83334 at iteration 98,000.
- Final training loss: 5.39471.
- Final validation loss: 4.90570.
- Achieved a new all-time best validation loss.
- Improved upon V2.23 despite increasing the parameter count by only 18,632 (approximately 0.83%).
- Continued improving considerably later into training than V2.23.
- Generated text showed promising sentence structures and convincing Shakespearean dialogue fragments.
- Examples included "I pardon thy woe, good sir" and "I will not sin to be the queen".

**Conclusion:**  
Adding a second hidden layer improved validation performance and produced promising generation results. The experiment demonstrated that increasing network depth could be more beneficial than simply increasing context length. V2.25 became the new best-performing model, although further generation comparisons are needed to establish whether grammatical quality and long-term coherence improved consistently.

---

## V2.26

**Configuration:**
- Architecture: Two-hidden-layer weight-tied N-gram with output projection
- Embedding dimension: 72
- Hidden dimension 1: 136
- Hidden dimension 2: 160
- Context length: 6
- Dropout: 0.35
- Batch size: 32
- Learning rate: 0.0007
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Cosine Annealing (T_max=100,000, eta_min=0.00005)
- Label smoothing: 0.1
- Training iterations: 100,000
- Validation batches: 500
- Parameters: 2,260,608

**Hypothesis:**  
Increasing the second hidden layer from 136 to 160 neurons could provide additional representational capacity, potentially improving next-token prediction and generated text quality while maintaining the existing architecture.

**Observations:**
- Best validation loss: 4.82736 at iteration 100,000.
- Final training loss: 5.39060.
- Final validation loss: 4.82736.
- Achieved a new all-time best validation loss.
- Improved upon V2.25, reducing validation loss from 4.83334 to 4.82736.
- Best validation loss was achieved at the final training iteration.
- Generated text contained promising Shakespearean expressions and dialogue-like structures.
- Notable phrases included "I'll have the father of winter", "I can warrant your mother" and "To give the destiny, love for a name."
- Hall of Fame: "I'll have the father of winter."

**Conclusion:**
Increasing the second hidden layer to 160 neurons produced a modest improvement over V2.25 and established a new validation-loss record. The experiment suggested that increasing network capacity could still provide benefits, although the improvement was relatively small.

---

## V2.27

**Configuration:**
- Architecture: Two-hidden-layer weight-tied N-gram with output projection
- Embedding dimension: 72
- Hidden dimension 1: 136
- Hidden dimension 2: 192
- Context length: 6
- Dropout: 0.35
- Batch size: 32
- Learning rate: 0.0007
- Optimizer: AdamW
- Weight decay: 0.01
- Scheduler: Cosine Annealing (T_max=150,000, eta_min=0.00005)
- Label smoothing: 0.1
- Training iterations: 150,000
- Validation batches: 500
- Parameters: 2,267,296

**Hypothesis:**  
Increasing the second hidden layer from 160 to 192 neurons could give the model additional capacity to process token representations and potentially improve next-token prediction without substantially increasing the parameter count.

**Observations:**
- Best validation loss: 4.81681 at iteration 107,000 (original run).
- Final training loss: 5.35344 at iteration 110,000 (original run).
- Final validation loss: 4.85914 at iteration 110,000 (original run).
- Second training run completed all 150,000 iterations.
- Second run's best validation loss: 4.83647 at iteration 130,000.
- Second run's final training loss: 5.28974.
- Second run's final validation loss: 4.82749.
- Second run did not improve upon the original checkpoint, which was preserved.
- Generated text showed recognisable Shakespearean expressions, character labels and dialogue-like structures.

**Conclusion:**  
Increasing the second hidden layer to 192 neurons produced a new best validation loss of 4.81681, outperforming V2.26. A second full training run came very close to reproducing this result but did not surpass it. V2.27 demonstrated consistent performance in the 4.8 validation-loss range.

---
## V2.28

**Configuration:**
- Architecture: Three-hidden-layer, weight-tied N-gram neural network
- Embedding dimension: 72
- Hidden layers: 136 → 192 → 192
- Context length: 6
- Dropout: 0.35
- Batch size: 32
- Optimiser: AdamW
- Learning rate: 0.0007
- Weight decay: 0.01
- Scheduler: CosineAnnealingLR
- T_max: 150,000
- Minimum learning rate: 0.00005
- Activation: ReLU
- Label smoothing: 0.1
- Training iterations: 150,000
- Validation batches: 500
- Parameters: 2,304,352

**Hypothesis:**

Introduce a third hidden layer to investigate whether additional depth improves the model's ability to learn complex relationships between token representations.

**Observations:**
- Best validation loss: 4.86010 at 105,000 iterations.
- Final training loss: 5.34381.
- Final validation loss: 4.89767.
- Did not outperform V2.27 (4.81681).
- Generated text maintained Shakespearean vocabulary and dialogue formatting, with some recognisable phrases but inconsistent coherence.

**Conclusion:**

Adding a third hidden layer increased model capacity but did not improve validation loss. Increasing depth alone was not sufficient to outperform the existing architecture.

---

## V2.29

**Configuration:**
- Architecture: Three-hidden-layer, weight-tied N-gram neural network
- Embedding dimension: 72
- Hidden layers: 136 → 192 → 224
- Context length: 6
- Dropout: 0.35
- Batch size: 32
- Optimiser: AdamW
- Learning rate: 0.0007
- Weight decay: 0.01
- Scheduler: CosineAnnealingLR
- T_max: 150,000
- Minimum learning rate: 0.00005
- Activation: ReLU
- Label smoothing: 0.1
- Training iterations: 150,000
- Validation batches: 500
- Parameters: 2,312,832

**Hypothesis:**

Increase the width of the third hidden layer from 192 to 224 neurons to investigate whether additional capacity improves the model's ability to combine learned representations.

**Observations:**
- Best validation loss: 4.83364 at 122,000 iterations.
- Final training loss: 5.33219.
- Final validation loss: 4.87742.
- Improved over V2.28 by approximately 0.02646.
- Came close to V2.27 but did not beat its validation loss.
- Generated text contained some convincing short expressions, including "It's my agony" and "I have no hope!"

**Conclusion:**

Increasing the third hidden layer's width improved validation performance. However, the model still did not outperform V2.27. This suggested that activation functions might be worth investigating next.

---

## V2.30

**Configuration:**
- Architecture: Three-hidden-layer, weight-tied N-gram neural network
- Embedding dimension: 72
- Hidden layers: 136 → 192 → 224
- Context length: 6
- Dropout: 0.35
- Batch size: 32
- Optimiser: AdamW
- Learning rate: 0.0007
- Weight decay: 0.01
- Scheduler: CosineAnnealingLR
- T_max: 150,000
- Minimum learning rate: 0.00005
- Activation: GELU
- Label smoothing: 0.1
- Training iterations: 150,000
- Validation batches: 500
- Parameters: 2,312,832

**Hypothesis:**

Replace ReLU with GELU while keeping the architecture and training configuration unchanged to investigate whether a smoother activation function improves learning and generalisation.

**Observations:**
- Best validation loss: 4.78941 at 122,000 iterations.
- Final training loss: 5.19158.
- Final validation loss: 4.84361.
- Improved over V2.29 by approximately 0.04423.
- Beat the previous all-time best, V2.27 (4.81681).
- Finished with a noticeably lower training loss than V2.29.
- Generated text showed improvements in dialogue, short phrases and overall coherence.

**Conclusion:**

GELU produced a significant improvement over ReLU in this architecture and established a new all-time best validation loss. This made activation functions a promising area for further experimentation.

---

## V2.31

**Configuration:**
- Architecture: Three-hidden-layer, weight-tied N-gram neural network
- Embedding dimension: 72
- Hidden layers: 136 → 192 → 224
- Context length: 6
- Dropout: 0.35
- Batch size: 32
- Optimiser: AdamW
- Learning rate: 0.0007
- Weight decay: 0.01
- Scheduler: CosineAnnealingLR
- T_max: 150,000
- Minimum learning rate: 0.00005
- Activation: SiLU
- Label smoothing: 0.1
- Training iterations: 150,000
- Validation batches: 500
- Parameters: 2,312,832

**Hypothesis:**

Replace GELU with SiLU while keeping everything else unchanged to investigate whether SiLU's different smooth activation curve can improve performance further.

**Observations:**
- Best validation loss: 4.78443 at 122,000 iterations.
- Final training loss: 5.16676.
- Final validation loss: 4.848
---

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
- "Clown: Hail, my poor – which ever souls"
- "GLOUCESTER: And do no more corn."
- "So me so slain."
- "Good lord:"
- "I will no Come is all"
- "Nurse: I cannot warrant."
- "I mickle you were all, which did of the heavens of Corioli"
- "The sink at to lay grace and"
- "A fatal page to make?"
- "So, I have a foes of my subjects"
- "What shall men yield."
- "My lord."
- "I pardon thy woe, good sir."
- "I'll have the father of winter."
- "And I am the fault of Thomas"
- "The hours of his of his friend."
- "It's my agony."
- "No, my lord."
- "Alack, and so, convey you: my gracious lord."
---

*FelixGPT — Teaching a computer to speak Shakespeare, one questionable sentence at a time.*