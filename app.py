import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFPlumberLoader, \
    UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# -----------------------------
# Logging Konfiguration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -----------------------------
# Erweiterte Konfiguration
# -----------------------------
@dataclass
class Config:
    model_name: str = "llama3.2"
    data_folder: str = "data"
    db_dir: str = "db"
    config_file: str = "config.json"
    history_file: str = "conversation_history.json"
    k: int = 3
    chunk_size: int = 1000
    chunk_overlap: int = 200
    temperature: float = 0.1
    max_tokens: int = 2000

    # Erweiterte Einstellungen
    similarity_threshold: float = 0.7
    enable_history: bool = True
    auto_save: bool = True
    supported_file_types: List[str] = field(default_factory=lambda: ['.txt', '.pdf', '.md', '.json'])


# -----------------------------
# Verbesserte Persönliche KI Klasse
# -----------------------------
class EnhancedPersonalAI:
    def __init__(self, config: Config):
        self.config = config
        self.llm = None
        self.emb = None
        self.db = None
        self.qa_chain = None
        self.conversation_history = []

        self._initialize_components()

    def _initialize_components(self):
        """Initialisiert alle Komponenten mit Fehlerbehandlung"""
        try:
            # Konfiguration laden
            self._load_config()

            # LLM & Embeddings initialisieren
            logger.info("Initialisiere Ollama LLM...")
            self.llm = OllamaLLM(
                model=self.config.model_name,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens
            )

            logger.info("Initialisiere Ollama Embeddings...")
            self.emb = OllamaEmbeddings(model=self.config.model_name)

            # Vector Store initialisieren
            logger.info("Initialisiere Chroma DB...")
            self._initialize_vector_store()

            # Retrieval Chain mit custom Prompt
            logger.info("Erstelle Retrieval Chain...")
            self._create_qa_chain()

            # Konversationshistorie laden
            if self.config.enable_history:
                self._load_conversation_history()

            logger.info("✅ KI erfolgreich initialisiert!")

        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung: {e}")
            raise

    def _load_config(self):
        """Lädt Konfiguration aus Datei oder verwendet Defaults"""
        if os.path.exists(self.config.config_file):
            try:
                with open(self.config.config_file, "r", encoding="utf-8") as f:
                    user_config = json.load(f)

                # Update Konfiguration mit Benutzereinstellungen
                for key, value in user_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)

                logger.info("Konfiguration geladen")
            except Exception as e:
                logger.warning(f"Konfigurationsdatei konnte nicht geladen werden: {e}")

    def _initialize_vector_store(self):
        """Initialisiert den Vector Store mit erweiterter Dokumentenverarbeitung"""
        # Prüfe ob DB existiert, sonst erstelle sie
        if os.path.exists(self.config.db_dir) and len(os.listdir(self.config.db_dir)) > 0:
            self.db = Chroma(
                persist_directory=self.config.db_dir,
                embedding_function=self.emb
            )
            logger.info("Existierende Chroma DB geladen")
        else:
            # Erstelle neue DB mit Dokumenten
            self.db = self._create_new_vector_store()

        # Immer Dokumente indexieren für Updates
        self._ensure_index()

    def _create_new_vector_store(self):
        """Erstellt einen neuen Vector Store"""
        logger.info("Erstelle neue Chroma DB...")
        return Chroma(
            persist_directory=self.config.db_dir,
            embedding_function=self.emb
        )

    def _get_document_loader(self):
        """Erstellt einen Document Loader für verschiedene Dateitypen"""
        loaders = {
            '.txt': TextLoader,
            '.pdf': PDFPlumberLoader,
            '.md': UnstructuredMarkdownLoader,
            '.json': TextLoader
        }

        return DirectoryLoader(
            self.config.data_folder,
            glob="**/*.*",
            loader_cls=lambda path: loaders.get(Path(path).suffix.lower(), TextLoader)(path),
            show_progress=True,
            use_multithreading=True
        )

    def _ensure_index(self):
        """Stellt sicher, dass alle Dokumente indexiert sind"""
        if not os.path.exists(self.config.data_folder):
            os.makedirs(self.config.data_folder)
            logger.warning(f"Datenordner {self.config.data_folder} erstellt - bitte Dokumente hinzufügen")
            return

        try:
            loader = self._get_document_loader()
            docs = loader.load()

            if len(docs) > 0:
                logger.info(f"Verarbeite {len(docs)} Dokumente...")

                # Dokumente aufteilen für bessere Verarbeitung
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                split_docs = text_splitter.split_documents(docs)

                logger.info(f"Indexiere {len(split_docs)} Text-Chunks...")

                # Zur Datenbank hinzufügen
                self.db.add_documents(split_docs)
                self.db.persist()

                logger.info("✅ Dokumente erfolgreich indexiert")
            else:
                logger.warning("Keine Dokumente zum Indexieren gefunden")

        except Exception as e:
            logger.error(f"Fehler beim Indexieren: {e}")

    def _create_qa_chain(self):
        """Erstellt die QA Chain mit custom Prompt"""
        # Custom Prompt für bessere Antworten
        custom_prompt = PromptTemplate(
            template="""Du bist ein hilfreicher, freundlicher Assistent. Beantworte die Frage basierend auf dem bereitgestellten Kontext.

Kontext: {context}

Frage: {question}

Antworte in derselben Sprache wie die Frage. Sei präzise und hilfreich. Wenn du die Antwort nicht im Kontext findest, sage ehrlich, dass du es nicht weißt.

Antwort:""",
            input_variables=["context", "question"]
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": self.config.k,
                    "score_threshold": self.config.similarity_threshold
                }
            ),
            chain_type_kwargs={"prompt": custom_prompt},
            return_source_documents=True
        )

    def ask(self, question: str) -> str:
        """Frage an die KI stellen mit erweiterter Funktionalität"""
        if not question.strip():
            return "Bitte stelle eine gültige Frage."

        try:
            logger.info(f"Verarbeite Frage: {question}")

            # Frage verarbeiten
            response = self.qa_chain.invoke({"query": question})
            answer = response.get("result", "Ich konnte keine Antwort generieren.")
            source_docs = response.get("source_documents", [])

            # Konversationshistorie aktualisieren
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "answer": answer,
                "sources_used": len(source_docs)
            }

            self.conversation_history.append(conversation_entry)

            # Auto-Save
            if self.config.auto_save:
                self._save_conversation_history()

            # Quellen hinzufügen falls verfügbar
            if source_docs:
                source_info = f"\n\n📚 Verwendete {len(source_docs)} Quellen"
                answer += source_info

            return answer

        except Exception as e:
            logger.error(f"Fehler bei der Frageverarbeitung: {e}")
            return "Entschuldigung, ein Fehler ist aufgetreten. Bitte versuche es erneut."

    async def ask_async(self, question: str) -> str:
        """Asynchrone Version der Frageverarbeitung"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.qa_chain.invoke({"query": question})
            )
            return response.get("result", "Keine Antwort verfügbar")
        except Exception as e:
            logger.error(f"Fehler bei asynchroner Verarbeitung: {e}")
            return "Entschuldigung, ein Fehler ist aufgetreten"

    def _load_conversation_history(self):
        """Lädt die Konversationshistorie"""
        try:
            if os.path.exists(self.config.history_file):
                with open(self.config.history_file, "r", encoding="utf-8") as f:
                    self.conversation_history = json.load(f)
                logger.info(f"Konversationshistorie mit {len(self.conversation_history)} Einträgen geladen")
        except Exception as e:
            logger.warning(f"Konversationshistorie konnte nicht geladen werden: {e}")

    def _save_conversation_history(self):
        """Speichert die Konversationshistorie"""
        try:
            with open(self.config.history_file, "w", encoding="utf-8") as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Konversationshistorie konnte nicht gespeichert werden: {e}")

    def get_conversation_history(self, limit: int = 10) -> List[Dict]:
        """Gibt die letzte Konversationshistorie zurück"""
        return self.conversation_history[-limit:] if self.conversation_history else []

    def clear_conversation_history(self):
        """Löscht die Konversationshistorie"""
        self.conversation_history.clear()
        if os.path.exists(self.config.history_file):
            os.remove(self.config.history_file)
        logger.info("Konversationshistorie gelöscht")

    def get_stats(self) -> Dict[str, Any]:
        """Gibt Statistiken über die KI zurück"""
        try:
            collection = self.db.get()
            doc_count = len(collection['documents']) if collection['documents'] else 0

            return {
                "model": self.config.model_name,
                "documents_indexed": doc_count,
                "conversation_entries": len(self.conversation_history),
                "retrieval_k": self.config.k,
                "chunk_size": self.config.chunk_size,
                "data_folder": self.config.data_folder
            }
        except Exception as e:
            logger.error(f"Fehler beim Sammeln von Statistiken: {e}")
            return {"error": "Statistiken nicht verfügbar"}

    def add_documents(self, file_paths: List[str]):
        """Fügt spezifische Dokumente zur Datenbank hinzu"""
        try:
            from langchain_community.document_loaders import UnstructuredFileLoader

            docs = []
            for file_path in file_paths:
                if os.path.exists(file_path):
                    loader = UnstructuredFileLoader(file_path)
                    docs.extend(loader.load())

            if docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                split_docs = text_splitter.split_documents(docs)
                self.db.add_documents(split_docs)
                self.db.persist()
                logger.info(f"{len(split_docs)} neue Dokument-Chunks hinzugefügt")
            else:
                logger.warning("Keine gültigen Dokumente gefunden")

        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Dokumenten: {e}")


# -----------------------------
# Erweiterte Main Funktion
# -----------------------------
def main():
    config = Config()
    ai = EnhancedPersonalAI(config)

    print("🚀 Erweiterte persönliche KI gestartet!")
    print("Befehle:")
    print("  'stats' - Zeige Statistiken")
    print("  'history' - Zeige Konversationsverlauf")
    print("  'clear' - Lösche Konversationsverlauf")
    print("  'reload' - Dokumente neu indexieren")
    print("  'exit' - Beenden")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n🧠 Deine Frage: ").strip()

            if user_input.lower() == 'exit':
                print("Auf Wiedersehen! 👋")
                break
            elif user_input.lower() == 'stats':
                stats = ai.get_stats()
                print("\n📊 Statistiken:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            elif user_input.lower() == 'history':
                history = ai.get_conversation_history(5)
                if history:
                    print("\n📝 Letzte Konversationen:")
                    for i, entry in enumerate(history, 1):
                        print(f"  {i}. {entry['question'][:50]}...")
                else:
                    print("Keine Konversationshistorie verfügbar.")
                continue
            elif user_input.lower() == 'clear':
                ai.clear_conversation_history()
                print("Konversationshistorie gelöscht.")
                continue
            elif user_input.lower() == 'reload':
                print("Dokumente werden neu indexiert...")
                ai._ensure_index()
                print("✅ Dokumente neu indexiert!")
                continue
            elif not user_input:
                continue

            # Normale Frage verarbeiten
            answer = ai.ask(user_input)
            print(f"\n💡 Antwort: {answer}")

        except KeyboardInterrupt:
            print("\n\nProgramm beendet.")
            break
        except Exception as e:
            logger.error(f"Unerwarteter Fehler: {e}")
            print("Ein Fehler ist aufgetreten. Bitte versuche es erneut.")


if __name__ == "__main__":
    main()