from transformers import  AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, render_template, request
import random
# tokenizer  = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
# model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# tokenizer.save_pretrained('saved_model')
# model.save_pretrained('saved_model')


port = 12345
def project_id():
    import json
    import os
    info = json.load(open(os.path.join(os.environ['HOME'], ".smc", "info.json"), 'r'))
    return info['project_id']

app = Flask(__name__)

model = AutoModelForCausalLM.from_pretrained("saved_model")

tokenizer =  AutoTokenizer.from_pretrained("saved_model")
chat_history_bot2human = None
chat_history_bot2bot = None
start = 0
set_of_questions=[ 'Can we go out together?', 'What is your color?', 'Are you a robot?', 'Tell me something', 'How can you help me?', 'What can you do?','Do you enjoy any music?', 'Do you have any favorite games?' , 'What do you think about yourself?', 'I enjoy hiking and the outdoors.','Can you make it to dinner tonight?', 'What is your best movie?',"What's one thing that can instantly make your day better?","In the summer, would you rather go to the beach or go camping?","Do you have any pet peeves?","Would you rather cook or order in?","What's your favorite board game?","Is there any product that you couldn't live without?","What type of role do you want to take on after this one?","What do you remember most about your first job?",
                 "What’s the worst job you’ve ever had?", "Do you believe in astrology?","Have you ever lost a friend?","What is your biggest irrational fear?"]
bot2_result = ""

@app.route("/")
def home():
    name = "ChatBot Home - Universal"
    return render_template('Home.html', name=name)


@app.route("/human2bot_get", methods=['POST', 'GET'])
def get_human2bot_response():
    userText = request.args.get('msg')
    return respond_bot2human(userText)


@app.route("/bot2bot_get", methods=['POST', 'GET'])
def get_bot2bot_response():
    return respond_bot2bot()

@app.route("/reset", methods=['POST', 'GET'])
def reset():
    global start
    global chat_history_bot2human
    start = 0
    chat_history_bot2human = None
    return ""
    
def respond_bot2human(question):
    global chat_history_bot2human
    new_question = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt' )
    if chat_history_bot2human is None:
        bot_input = new_question
    else:
        bot_input = torch.cat([chat_history_bot2human, new_question], dim=-1)
    chat_history_bot2human = model.generate(bot_input, max_length=100_000, pad_token_id=tokenizer.eos_token_id)
    
    return tokenizer.decode(chat_history_bot2human[:, bot_input.shape[-1]:][0], skip_special_tokens=True)

    
def respond_bot2bot():
    global start
    global set_of_questions
    global bot2_result
    global chat_history_bot2bot
    if start==0:
        bot1_result = random.choice(set_of_questions) 
    else:
        bot1_input = chat_history_bot2bot
        chat_history_bot2bot =  model.generate(bot1_input, max_length=100_000, pad_token_id=tokenizer.eos_token_id)[:, bot1_input.shape[-1]:]
        bot1_result = tokenizer.decode(chat_history_bot2bot[0], skip_special_tokens=True)
    
    if bot1_result == bot2_result:
        bot1_result = random.choice(set_of_questions)
        chat_history_bot2bot = tokenizer.encode(bot1_result + tokenizer.eos_token, return_tensors='pt' )
    
    if start==0:
        bot2_input = tokenizer.encode(bot1_result + tokenizer.eos_token, return_tensors='pt' )
        start+=1
    else:
        bot2_input = chat_history_bot2bot
        
    chat_history_bot2bot =  model.generate(bot2_input, max_length=100_000, pad_token_id=tokenizer.eos_token_id)[:, bot2_input.shape[-1]:]
    bot2_result = tokenizer.decode(chat_history_bot2bot[0], skip_special_tokens=True)

    if bot1_result == bot2_result:
        bot2_result = random.choice(set_of_questions)
        chat_history_bot2bot = tokenizer.encode(bot2_result + tokenizer.eos_token, return_tensors='pt' )
    return {"bot1":bot1_result,"bot2":bot2_result}


if __name__ == "__main__":
    # you will need to change code.ai-camp.org to other urls if you are not running on the coding center.
    print(f"Try to open\n\n    localhost:{port} \n\n")
    app.run(host = '0.0.0.0', port = port, debug=True)
    
    import sys; sys.exit(0)
