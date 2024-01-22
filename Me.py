from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import openai
from secret_key import Api_keys
import os 
os.environ['OPENAI_API_KEY']=Api_keys
llm=OpenAI(temperature=0)

AboutMe='''I am Muhammad Ahsan student of universty of engineering and technology Mardan
i am from Nowshera Reciently i complete the course of python and Supervised Machine learning from 
Coursera website Now i am working on the Langchain So i have Amazing experience in the 
Python and machine learning.
Overall, I am grateful for the experience and I am excited to see where my career takes me next,
{history}
Human: {input}
AI Assistant:'''

myprompt=PromptTemplate(
    input_variables=['history','input'],
    template=AboutMe
)

myconversation=ConversationChain(
    prompt=myprompt,
    llm=llm,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Ai assitant")
)
name="Muhammd Ahsan"
food_choices="kfc,foods"
fitness_level="50 percent "
daily_activity="5 time prayer"
mood="positive"
goals="improved AI skill"


# ai_response = myconversation.predict(input=f"""
# Here is user data:
# "user name is {name}, and their food choices include {food_choices}. 
# This user is actively pursuing a fitness journey, and they've provided insightful details:
# - Fitness Level: {fitness_level}
# - Daily Activity: {daily_activity}
# - Mood: {mood}

# Furthermore, they have set specific fitness goals, and their current
# goals are as follows: {goals}.

# # """)

# print(response)  
while True:
    user_input = input("You: ")
    ai_response = myconversation.predict(input=user_input)
    print(f"Ahsan: {ai_response}")
