# 🤖 pistoBot models
Python resource used to train different ML models to generate text learning from personal chats.


## 🎭 Models available:
- RNN vanilla - under `/01_RNN/`
    - Based on [tensorflow example](https://www.tensorflow.org/tutorials/text/text_generation)
- GPT-2 pre trained - under `/02_gpt2_simple/` 
    - Based on [`gpt-2-simple`](https://github.com/minimaxir/gpt-2-simple)
    
- GPT-2 trained from scratch - under `03_gpt2_scratch` folder
   - Based on [`aitextgen`](https://github.com/minimaxir/aitextgen)
   
**Requirements**<br>
- At project root the "common" requirements and each model have custom `requirements.txt` inside the folder.
- This code has only been tested with Python >= 3.6.
- Could use `./colab/run_training.sh` to launch training
