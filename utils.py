from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st

openai.api_key = "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" ## find at platform.openai.com
model = SentenceTransformer('all-MiniLM-L6-v2')

pinecone.init(api_key='4f4a45d8-d09e-4e12-b1b2-0ab0fba4a851', # find at app.pinecone.io
              environment='us-east1-gcp' # next to api key in console
             )
# index = pinecone.Index('' # index name from pinecone)
index = pinecone.Index('ce322module2')


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=2, includeMetadata=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
