
from utilities import gloss_list_to_speech
from typing import List, Any 
from langchain.prompts import PromptTemplate 
from langchain.chains import LLMChain 
from langchain_google_genai import ChatGoogleGenerativeAI
from gtts import gTTS
from dotenv import load_dotenv
from dotenv import load_dotenv
import os
# Load the environment variables from .env file
load_dotenv()
try:
    gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", #Call ListModels to see the list of available models and their supported methods
                                    google_api_key=os.environ.get("API_KEY"))
except Exception as e:
    print(f"Errore durante l'inizializzazione di ChatGoogleGenerativeAI: {e}")
    print("Assicurati di aver impostato la variabile d'ambiente GOOGLE_API_KEY.")

gemini_template_str = """Sei un traduttore professionista della Lingua dei Segni Italiana (LIS) che lavora con gloss italiani.
Ricevi una sequenza di **gloss scritti in italiano**, ciascuna delimitata dal simbolo pipeline (|).
Ogni gloss rappresenta una parola o una frase **(in italiano)** che esprime il significato di un segno della LIS.
Il tuo compito è convertire questa sequenza di gloss in una **frase in lingua italiana** ben strutturata e concisa, facilmente comprensibile da chi non conosce la lingua dei segni.
Assicurati che la frase sia grammaticalmente corretta e abbia un tono naturale.
Importante: l'output non deve contenere il delimitatore pipeline (|).

Ecco alcuni esempi:
input: 'tu | nome | quale | essere'
output: 'Qual è il tuo nome?'

input: 'tu | abitare | dove'
output: 'Dove abiti?'

Gloss da tradurre (in italiano):
{text}

Frase tradotta (in italiano):"""

# --- example ----
gloss_list_example = ["Tu", "Cosa", "Mangiare", "Cosa", "tu"]

if 'gemini_llm' in globals(): # Verify if gemini_llm has been initialized
    generated_audio = gloss_list_to_speech(gloss_list_example, gemini_llm, gemini_template_str)
else:
    print("LLM not initialized. Please check your environment variables and initialization code.")