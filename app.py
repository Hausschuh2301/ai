import os
import json
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
from google_search_results import GoogleSearchResults

# ------------------------
# KONFIGURATION
# ------------------------
MODEL_NAME = "llama3"
DATA_FOLDER = "./data"
DB_DIR = "./db"
CONFIG_FILE = "./config.json"
SERPAPI_KEY = "DEIN_SERPAPI_KEY_HIER"  # Trage hier deinen SerpAPI Key ein

# ------------------------
# PersonalAI Klasse
# ------------------------
class PersonalAI:
    def __init__(self, model_name, data_folder, db_dir, api_key):
        self.model_name = model_name
        self.data_folder = data_folder
        self.db_dir = db_dir
        self.api_key = api_key

        print("[INFO] Ollama LLM initialisieren...")
        self.llm = OllamaLLM(model=self.model_name)

        print("[INFO] Ollama Embeddings initialisieren...")
        self.emb = OllamaEmbeddings(model=self.model_name)

        print("[INFO] Chroma DB aufbauen/öffnen...")
        self.db = Chroma(persist_directory=self.db_dir, embedding_function=self.emb)

        self._ensure_db()

    def _ensure_db(self):
        # Lade Dokumente aus dem Datenordner
        loader = DirectoryLoader(self.data_folder)
        docs = loader.load()

        if not docs:
            print("[INFO] Leere DB - initialer Import wird gestartet.")
            self.db.add_documents(docs)
        else:
            print("[INFO] Chroma DB geladen (existierende Einträge vorhanden).")

        # Setup Retrieval Chain mit invoke()-Methode
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.db.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )

    def search_online(self, query):
        """Sucht online mit SerpAPI und gibt die Ergebnisse zurück."""
        params = {
            "q": query,
            "hl": "de",
            "gl": "de",
            "api_key": self.api_key
        }
        search = GoogleSearchResults(params)
        results = search.get_dict()
        hits = results.get("organic_results", [])
        return [(item.get("title"), item.get("link")) for item in hits[:5]]  # Top 5 Ergebnisse

    def ask(self, question):
        """Versuche zuerst lokal, falls nichts gefunden wird → online suchen."""
        # Lokale Suche mit invoke()
        try:
            output = self.qa_chain.invoke({"query": question})
            result_text = output.get("result", "").strip()
            if result_text:
                return result_text
        except Exception as e:
            print(f"[WARN] Fehler bei lokaler Suche: {e}")

        # Online-Suche, falls lokal nichts gefunden
        online_results = self.search_online(question)
        if online_results:
            response = "Ich konnte keine lokale Antwort finden. Hier sind einige Online-Ergebnisse:\n"
            for i, (title, link) in enumerate(online_results, start=1):
                response += f"{i}. {title} - {link}\n"
            return response
        else:
            return "Keine Antwort gefunden, weder lokal noch online."

# ------------------------
# MAIN
# ------------------------
def main():
    ai = PersonalAI(
        model_name=MODEL_NAME,
        data_folder=DATA_FOLDER,
        db_dir=DB_DIR,
        api_key=SERPAPI_KEY
    )

    print("\nDeine persönliche KI läuft. Tippe 'exit' zum Beenden.\n")
    while True:
        question = input("Deine Frage: ").strip()
        if question.lower() == "exit":
            break
        answer = ai.ask(question)
        print("\nAntwort:\n", answer, "\n")

if __name__ == "__main__":
    main()
