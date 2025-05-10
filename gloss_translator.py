from utilities import gloss_list_to_speech
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

class GlossTranslator:
    def __init__(self):
        load_dotenv()
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=os.environ.get("API_KEY")
            )
        except Exception as e:
            print(f"Errore durante l'inizializzazione di ChatGoogleGenerativeAI: {e}")
            self.llm = None

        self.template = """Sei un traduttore professionista della Lingua dei Segni Italiana (LIS) che lavora con gloss italiani.
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

    def translate_and_speak(self, gloss_list):
        if self.llm:
            return gloss_list_to_speech(gloss_list, self.llm, self.template)
        else:
            raise RuntimeError("LLM non inizializzato. Controlla le variabili d'ambiente.")



if __name__ == "__main__":
    gloss_list_example = ["Io", "Cosa", "Mangiare", "Cosa", "Io", "Fare"]
    translator = GlossTranslator()
    try:
        audio_result = translator.translate_and_speak(gloss_list_example)
    except RuntimeError as e:
        print(e)
