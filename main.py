import json
import os

import dotenv
import langchain
import langchain.chains
import langchain.chat_models
import langchain.llms
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
