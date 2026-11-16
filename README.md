# Custom LLM from scratch - FelixGPT
A project where I make a custom LLM/GPT from scratch, everything, with help from ChatGPT to learn.
---
You can view detailed notes about each model type in their respective PROGRESS.md files, currently there is only one in the Shakespeare folder under the felixgpt_Xgrams folder.
---
**DISCLAIMER:**

I did not make everything myself. In the earlier files such as the bigram ones I had a lot of help from ChatGPT. It coded parts, explained concepts and the math. Even now ChatGPT writes some smaller parts but with the newer Ngram versions I aim to write a lot myself and have ChatGPT check it, then debug together and fix it until it looks right.

My learning strategy is in this direction: It starts with ChatGPT writing a lot more than me and explaining, then (currently) I write more and larger parts myself with ChatGPT checking it as well as providing parts of new or complex aspects. Eventually I aim to be able to write substantial parts myself with ChatGPT only checking it roughly.

Furthermore, FelixGPT is an experimental educational project. Its models are not intended to compete with modern commercial language models. Their primary purpose is to explore machine learning concepts and demonstrate the development of language models through experimentation.
---

# FelixGPT

**A project where I make my own language model and eventually a GPT**

FelixGPT is a personal machine learning project to see how language models work, how they can be built, and how different architectures and training approaches affect their performance.

The project started with a simple bigram model, which predicted the next character based on the previous one. What began as an experiment gradually evolved into more advanced neural language models, eventually leading to the development of word-level N-gram models.

Rather than relying on existing pretrained models, the goal is to understand and implement the underlying concepts myself, experimenting with different architectures, tokenisation methods, hyperparameters and training techniques along the way.

In the future I still want to advance on these N-gram models, located under felixgpt_Xgrams. After Shakespeare models yield the best possible, or very good, results I want to make conversational versions. I plan to add self attention and make a Transformer in the future too. 

## Development

The project currently focuses on experimental N-gram language models trained on Tiny Shakespeare.

Development has included:

- **Bigram model:** The original experiment that started the project, exploring character-level next-token prediction.
- **Character-level models:** Early neural language models trained on individual characters.
- **Word-level N-gram models:** Models using word-level tokenisation and fixed-length context windows.
- **Hyperparameter experiments:** Testing different embedding dimensions, hidden widths, context lengths, dropout rates and other training parameters.
- **Tokenisation:** Experimenting with different regex-based tokenisation approaches, including GPT-2 style pre-tokenisation using its regex.
- **Model evaluation:** Comparing training and validation losses, generated samples and overall generation quality.

The latest experiments include V2.15, which introduced a revised tokenisation approach and improved generation quality compared with earlier versions.

## Future Development

FelixGPT is an ongoing project, with several areas planned for further exploration:

- Further improvements to the existing N-gram models.
- Experiments with fine-tuning and transfer learning.
- Development of custom Transformer architectures under the ION architecture concept (naming in progress, if possible, ION would stand for Intelligence Oriented Network).
- Larger language models with increased parameter counts, supported by larger and more sophisticated datasets.
- More advanced tokenisation and training techniques.
- Continued experimentation with model performance and generation quality.

The long-term goal is to develop increasingly capable language models while gaining a deeper understanding of the technologies behind them.

## Inspiration and Contributions

This project is intended to be a learning experience, and I'm happy for others to explore the code, learn from the experiments and take inspiration from the ideas presented here.

You are welcome to study the implementation and develop your own projects inspired by this work, provided you respect the terms of the project's licence.

Please refer to the `LICENSE` file before using, modifying or redistributing any part of this repository.

---
