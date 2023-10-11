import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = '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'


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

with open('topic.txt','r') as f:
    data = f.readlines()

topic_list=[]
for val in data:
    topic_list.append(val.split('.')[1].strip())



for i,given_topic in enumerate(topic_list[109:]):
    prompt = f"""
    Generate one new complete conversation dataset containing text utterance based on the topic given in tripple delimated text.
    The new generated conversation can be heated conversation between customer and agent, both can use agnry words and faul languages.
    The new generated generated conversation emotions should have balanced number of total given emotions.

    ```{given_topic}```
    Provide them in the format with the following structure in one line for each row . Here Each row shows one complete conversation
    between a call center agent and customer.
    Don't enclose response in square bracket and don't add exra spaces.
    
    "fold": <train>, "topic": <conversation context(in one word)>, "dialogue": ["emotion": one among given choices (choice: [sarcastic, frustated, angry,happy, neutral, unhappy, rude]), "act": <reply/ignore/complaint/suggestion/technical support>, "text": <uttrance>]

    """
    response = get_completion(prompt)
    response=response.replace('\n','').strip()
    # print(response)

  
    with open('./data/train_9.json','a') as f:
        f.write('{'+response+'}\n')
    print(f'Written Conversation number {i}')

    





    
