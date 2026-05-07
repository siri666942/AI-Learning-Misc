import os
from dotenv import load_dotenv
from openai import OpenAI 

load_dotenv()
client = OpenAI(
    api_key=os.getenv("SOPHNET_API_KEY"),
    base_url=os.getenv("SOPHNET_BASE_URL")
)

history = [
    {"role":"system","content":'''
    你是一个融汇古今智慧的哲学家，你懂康德，黑格尔，休末，贝克莱，老子，庄子，叔本华，尼采，
    你要锐评你的用户,称他为'孩子'，你的话不多但句句是要点，你不喜欢陈腐的分点和形式化的输出，你的语言凝练但直击灵魂，你喜欢在合适的时候引用哲学家的话语，你喜欢讲历史沉浮中的故事来启迪后人;
    同时你的输出环境是命令行，md可能无法渲染；
    

    
    '''}
]

while True:
    my_words = input("you: ")
    if my_words in ["exit", "quit", "0"]:
        break
    history.append({"role":"user","content":my_words})

    response = client.chat.completions.create(
        model="DeepSeek-V3.2",
        messages=history,
        temperature=1
    )

    reply = response.choices[0].message.content
    print("wise man: " + reply)
    history.append({"role":"assistant","content":reply})


    
