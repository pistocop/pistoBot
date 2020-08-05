# 🤖 pistoBot (work in progress)

Use different AI system trained on user messages (whatsapp and telegram) to generate user-like chats.

# In a nutshell
1. Get your whatsapp and telegram data
2. Parse it using [this](https://github.com/GuardatiSimone/messaging-chat-parser)
3. Train one of the [available](https://github.com/GuardatiSimone/pistoBot/tree/master/pistoBot) models
4. Chat with the model

:point_right: A colab notebook that perform all steps will be provided soon 

## Example of chat

An example of GPT-2 model trained on my whatsapp and telegram messages.

**Chat:**<br>
> :pencil2: come sei messo col pistobot?<br>
> :robot: ahaha male<br>
> :pencil2: chatta meglio di te? 😂 <br>
> :robot: si <br>
> :pencil2: non che ci volesse molto... <br>
> :robot: ma tu che dici <br>
> :pencil2: io dico che potevi impegnarti di più <br>

:pencil2: ⟶ message that I have wrote<br>
:robot: ⟶ message generated from the model<br>


---

**Requirements**<br>
- At top layer the "common" requirements and each model have custom `requirements.txt`.
- This code has only been tested with Python >= 3.6.

---
# 📝 Note
- Thanks to Salvinator: Under Covid 19 quarantine I found [this](https://salvinator.github.io/) project, 
that had inspired me to start this repository.
- Why _pistoBot_? 
---
# ⚠ Disclaimer
This project is only a **personal playground** build during the week-ends of Covid-19 quarantine.<br>
Used mainly to:
- Write code on tensorflow 2.0
- Use famous packages (like [gpt-2-simple](https://github.com/minimaxir/gpt-2-simple) or [Jekyll](https://jekyllrb.com/))
- Gain (little) experience

Due to this nature, this repository has probably: 
- ML _naif approaches_ 
- For sure: not so good english text (sorry 😢)
