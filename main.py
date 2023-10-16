import json
import os

import dotenv
import langchain
import langchain.chains
import langchain.chat_models
import langchain.llms
import langchain.memory
import langchain.prompts


# Load environment variables from .env file
dotenv.load_dotenv()

# Get the OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")


################################################################
## EXAMPLE 1: Generate a SINGLE response with the GPT-3 engine #
################################################################

# Initialize the OpenAI object
llm = langchain.llms.OpenAI(
    openai_api_key=openai_api_key, model="gpt-3.5-turbo-instruct-0914"
)

# Define the prompt template
template = """Question: {question}

Answer: """
prompt = langchain.prompts.PromptTemplate(
    template=template, input_variables=["question"]
)

# user question
question = "What is the capital of France?"

llm_chain = langchain.chains.LLMChain(prompt=prompt, llm=llm)

response = llm_chain.run(question=question)

print(response)


##################################################################################
## EXAMPLE 1B: Generate a SINGLE response with the GPT-4 (chat-optimized) engine #
##################################################################################

# Initialize the OpenAI object:
# "langchain.chat_models.ChatOpenAI" replaces "langchain.llms.OpenAI"
# for access to GPT-4 and other chat-optimized models

llm = langchain.chat_models.ChatOpenAI(openai_api_key=openai_api_key, model="gpt-4")

# Reusing template, prompt and question from EXAMPLE 1

llm_chain = langchain.chains.LLMChain(prompt=prompt, llm=llm)

response = llm_chain.run(question=question)

print(response)


#################################################################
## EXAMPLE 2: Generate MULTIPLE responses with the GPT-3 engine #
#################################################################

# Initialize the OpenAI object
llm = langchain.llms.OpenAI(openai_api_key=openai_api_key, model="text-davinci-003")

# Define the prompt template
template = """Question: What is the capital of {country}?

Answer (as a JSON record containing the country and the capital, as {{ "country": ..., "capital": ... }} and nothing more):"""
prompt = langchain.prompts.PromptTemplate(
    template=template, input_variables=["country"]
)

# user questions: list of records qith country: "..."
questions = [
    {"country": "France"},
    {"country": "Germany"},
    {"country": "Spain"},
    {"country": "Italy"},
    {"country": "United Kingdom"},
]

llm_chain = langchain.chains.LLMChain(prompt=prompt, llm=llm)

response = llm_chain.generate(questions)

response_text = [
    json.loads(blurb.text.strip())
    for generation in response.generations
    for blurb in generation
]

print(response_text)


###################################################################
## EXAMPLE 3: Use a Chat model with history but no knowledge base #
###################################################################

llm = langchain.chat_models.ChatOpenAI(model_name="gpt-4", temperature=0)

# initialize the conversation history buffer — currently stores everything
memory = langchain.memory.ConversationBufferMemory()

# alternatively, you can use a memory that stores only the last 1000 tokens
# memory = langchain.memory.ConversationBufferMemory(max_tokens=1000)

# alternatively, you can use a memory that stores only the last k interactions
memory = langchain.memory.ConversationBufferWindowMemory(k=4)

# alternatively, you can use a memory that summarizes the conversation
# memory = langchain.memory.ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)

# see following for more information:
# https://www.pinecone.io/learn/series/langchain/langchain-conversational-memory/

# provide context to the memory
memory.save_context(
    {"input": "Hello my name is Skylar."},
    {
        "output": "Thanks for letting me know, I will remember that in future interactions."
    },
)
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = langchain.memory.ConversationBufferMemory()
memory.ai_prefix = "AI"
memory.human_prefix = "Human"
memory.save_context(
    {"input": "Hello, my name is Skylar."}, {"output": f"I will remember this."}
)
memory.save_context(
    {"input": "What is on the schedule today?"}, {"output": f"{schedule}"}
)

# Define the prompt template

template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""
prompt = langchain.prompts.PromptTemplate(
    template=template, input_variables=["history", "input"]
)

conversation = langchain.chains.ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
)


def speak(msg):
    print(">>>", msg)
    output = conversation.predict(input=msg)
    print("<<<", output)
    return output


speak("What is my name?")
speak("What would be a good time for a coffee break?")


###########################################################################################
## EXAMPLE 3B: Use a Chat model with history but no knowledge base, and tweaked template ##
###########################################################################################

llm = langchain.chat_models.ChatOpenAI(model_name="gpt-4", temperature=0.3)

# Define the prompt template

template = """This is a REPL session between a user and a foundational model. The output should be a list of 1 or more questions in JSON format, each with at least 4 distractors, each with an explanation:
[{{"prompt":"This is a prompt $x^2$","answers":[{{"answer":"This is an answer","correct":true,"explanation":"Because ..."}},{{"answer":"This is an answer","correct":false,"explanation":"Because..."}},...]}}]

Current conversation:
{history}
Input: {input}
Output as JSON:"""
prompt = langchain.prompts.PromptTemplate(
    template=template, input_variables=["history", "input"]
)

# provide context to the memory
memory = langchain.memory.ConversationBufferMemory(prompt=prompt)
memory.ai_prefix = "Output as JSON"
memory.human_prefix = "Input"

conversation = langchain.chains.ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True,
    prompt=prompt,
)


def speak(msg):
    print(">>>", msg)
    output = conversation.predict(input=msg)
    print("<<<", output)
    return output


result = speak("Generate a question of 4-th grade trigonometry.")
print(json.dumps(json.loads(result), indent=2))
