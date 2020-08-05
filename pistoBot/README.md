# 🤖 pistoBot
Python resource used to train different ML models to generate text learning from personal chats.

## 🖊 Personal chats
You can extract messages wrote by yourself using [this](https://github.com/GuardatiSimone/messaging-chat-parser) repository

## 🎭 Models available:
- RNN vanilla - under `/01_RNN/`
    - Based on official [tensorflow example](https://www.tensorflow.org/tutorials/text/text_generation)
- GPT-2 pre trained - under `/02_gpt2_simple/` 
    - Based on [`gpt-2-simple`](https://github.com/minimaxir/gpt-2-simple) package
    
- GPT-2 trained from scratch - under `03_gpt2_scratch` folder
   - Based on [aitextgen](https://github.com/minimaxir/aitextgen)
   