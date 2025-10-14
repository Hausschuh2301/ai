import os
import json
import logging
import asyncio
import requests
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------------
# KI Konfiguration
# -----------------------------
@dataclass
class AIConfig:
    model_name: str = "llama3.2"
    data_folder: str = "data"
    db_dir: str = "db"
    config_file: str = "config.json"
    history_file: str = "conversation_history.json"
    test_env_dir: str = "test_environment"

    # Einstellungen
    k: int = 6
    chunk_size: int = 1200
    chunk_overlap: int = 200
    temperature: float = 0.3
    max_tokens: int = 3000

    # Features
    enable_internet: bool = True
    enable_test_environment: bool = True
    enable_code_execution: bool = True
    enable_advanced_analysis: bool = True


# -----------------------------
# Internet-Suche
# -----------------------------
class InternetSearch:
    """Internet-Suche mit multiple Quellen"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info("Internet-Zugriff initialisiert")

    def web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Web-Suche mit multiple Quellen"""
        results = []

        # Mehrere Such-APIs kombinieren
        sources = [
            self._search_duckduckgo(query, max_results),
            self._search_wikipedia(query),
        ]

        for source_results in sources:
            results.extend(source_results)

        return results[:max_results]

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        """DuckDuckGo Suche"""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = []

                if data.get('Abstract'):
                    results.append({
                        'title': data.get('Heading', query),
                        'content': data['Abstract'],
                        'source': 'DuckDuckGo',
                        'url': data.get('AbstractURL', '')
                    })

                # Related Topics
                for topic in data.get('RelatedTopics', [])[:3]:
                    if 'Text' in topic:
                        results.append({
                            'title': topic.get('FirstURL', 'Topic').split('/')[-1],
                            'content': topic['Text'],
                            'source': 'DuckDuckGo',
                            'url': topic.get('FirstURL', '')
                        })

                return results
        except Exception as e:
            logger.warning(f"DuckDuckGo Suche fehlgeschlagen: {e}")

        return []

    def _search_wikipedia(self, query: str) -> List[Dict[str, str]]:
        """Wikipedia Suche"""
        try:
            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{query.replace(' ', '_')}"
            response = self.session.get(url, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return [{
                    'title': data.get('title', 'Wikipedia'),
                    'content': data.get('extract', ''),
                    'source': 'Wikipedia',
                    'url': data.get('content_urls', {}).get('desktop', {}).get('page', '')
                }]
        except Exception as e:
            logger.warning(f"Wikipedia Suche fehlgeschlagen: {e}")

        return []

    def scrape_website(self, url: str) -> Dict[str, str]:
        """Scraped Webseiten-Inhalte"""
        try:
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extrahiere Titel
                title = soup.find('title')
                title_text = title.text.strip() if title else "Kein Titel"

                # Extrahiere Hauptinhalt
                content_elements = soup.find_all(['p', 'h1', 'h2', 'h3'])[:8]
                content = ' '.join([elem.text.strip() for elem in content_elements if elem.text.strip()])

                return {
                    'title': title_text,
                    'content': content[:1000],
                    'url': url,
                    'source': 'Web Scraping',
                    'scraped_at': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Webseiten-Scraping fehlgeschlagen: {e}")

        return {}


# -----------------------------
# Test-Umgebung für Code
# -----------------------------
class TestEnvironment:
    """Test-Umgebung für Code-Ausführung"""

    def __init__(self, env_dir: str = "test_environment"):
        self.env_dir = Path(env_dir)
        self.env_dir.mkdir(exist_ok=True)

    def execute_python_code(self, code: str, timeout: int = 30) -> Dict[str, Any]:
        """Führt Python-Code aus"""
        try:
            # Erstelle temporäre Datei
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                temp_file = f.name

            # Führe Code aus
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.env_dir)
            )

            # Aufräumen
            os.unlink(temp_file)

            return {
                'success': result.returncode == 0,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'stdout': '',
                'stderr': 'Timeout: Code execution took too long',
                'returncode': -1
            }
        except Exception as e:
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Execution error: {str(e)}',
                'returncode': -1
            }


# -----------------------------
# Erweiterte Analyse
# -----------------------------
class AdvancedAnalysis:
    """Erweiterte Analyse mit Multi-Step Reasoning"""

    def __init__(self, llm):
        self.llm = llm

    def analyze_problem(self, question: str, max_steps: int = 3) -> str:
        """Analysiert Probleme mit Multi-Step Reasoning"""
        reasoning_steps = []

        for step in range(max_steps):
            if step == 0:
                prompt = f"""
Problemanalyse - Schritt {step + 1}:

Problem: {question}

Analysiere:
1. Was ist das Kernproblem?
2. Welche Informationen sind verfügbar?
3. Welche Lösungsansätze gibt es?
"""
            else:
                prompt = f"""
Weiterführende Analyse - Schritt {step + 1}:

Originalproblem: {question}
Bisherige Analyse: {reasoning_steps[-1]}

Nächste Schritte:
1. Vertiefe die Analyse
2. Identifiziere beste Lösung
3. Plane Umsetzung
"""

            response = self.llm.invoke(prompt)
            reasoning_steps.append(response.strip())

            # Prüfe ob Lösung gefunden
            if self._is_analysis_complete(response):
                break

        return self._compile_answer(question, reasoning_steps)

    def _is_analysis_complete(self, analysis: str) -> bool:
        """Prüft ob Analyse vollständig ist"""
        complete_indicators = ["zusammenfassung", "fazit", "ergebnis", "lösung:", "empfehlung:"]
        return any(indicator in analysis.lower() for indicator in complete_indicators)

    def _compile_answer(self, question: str, reasoning_steps: List[str]) -> str:
        """Kompiliert finale Antwort"""
        final_prompt = f"""
Erstelle eine umfassende Antwort:

Frage: {question}

Analyseschritte: {len(reasoning_steps)}

Erstelle eine strukturierte Antwort:
- Fasse die wichtigsten Erkenntnisse zusammen
- Gebe konkrete Handlungsempfehlungen
- Füge Code-Beispiele hinzu wenn relevant
- Erkläre die Lösung Schritt für Schritt

Antwort:
"""

        response = self.llm.invoke(final_prompt)
        return f"🔍 Detaillierte Analyse:\n\n{response.strip()}"


# -----------------------------
# KI Hauptklasse
# -----------------------------
class IntelligentAI:
    def __init__(self, config: AIConfig):
        self.config = config
        self.llm = None
        self.emb = None
        self.db = None
        self.qa_chain = None

        # Features
        self.internet_search = InternetSearch() if config.enable_internet else None
        self.test_env = TestEnvironment(config.test_env_dir) if config.enable_test_environment else None
        self.advanced_analysis = None

        self._initialize_ai()

    def _initialize_ai(self):
        """Initialisiert die KI"""
        try:
            logger.info("Initialisiere KI...")

            # Modell
            self.llm = OllamaLLM(
                model=self.config.model_name,
                temperature=self.config.temperature,
                num_predict=self.config.max_tokens
            )

            self.emb = OllamaEmbeddings(model=self.config.model_name)
            self._initialize_vector_store()
            self._create_qa_chain()

            # Fakten hinzufügen
            self.add_custom_facts()

            # Erweiterte Analyse
            if self.config.enable_advanced_analysis:
                self.advanced_analysis = AdvancedAnalysis(self.llm)

            logger.info("KI bereit!")
            self._show_capabilities()

        except Exception as e:
            logger.error(f"KI Initialisierung fehlgeschlagen: {e}")
            raise

    def add_custom_facts(self):
        """Fügt automatisch Fakten zur Datenbank hinzu"""
        custom_facts = [
            {
                "content": """
                PERSÖNLICHE KI-INFORMATIONEN:

                Ich bin eine intelligente KI, entwickelt als persönlicher Assistent.
                Meine Hauptfähigkeiten umfassen:
                - Natürlichsprachliche Kommunikation
                - Dokumenten-basierte Wissensverwaltung
                - Internet-Recherche und Web-Scraping
                - Code-Generierung und Ausführung
                - Komplexe Problemanalyse

                Technische Basis:
                - Entwickelt mit Python und LangChain
                - Verwendet Ollama für lokale Sprachmodelle
                - ChromaDB für Vektor-basierte Suche
                - Unterstützt multiple Dateiformate

                Besondere Features:
                - Echtzeit-Internet-Suche
                - Test-Umgebung für Code
                - Multi-Step Reasoning
                - Automatische Dokumentenverarbeitung
                """,
                "metadata": {"source": "system_facts", "type": "personal_info"}
            },
            {
                "content": """
                SYSTEMKONFIGURATION:

                Standard-Modell: llama3:8b
                Datenbank: ChromaDB mit persistenter Speicherung
                Such-Algorithmus: Similarity Search mit Embeddings
                Unterstützte Dateitypen: .txt, .pdf, .md, .json, .py

                Funktionsmodule:
                - InternetSearch: Web-Recherche und Scraping
                - TestEnvironment: Code-Ausführung und Testing
                - AdvancedAnalysis: Komplexe Problemlösung

                Entwicklungs-Stack:
                - LangChain für KI-Pipelines
                - Ollama Embeddings für Text-Vektorisierung
                - BeautifulSoup für Web-Scraping
                - Requests für HTTP-APIs
                """,
                "metadata": {"source": "system_config", "type": "technical_info"}
            }
        ]

        try:
            documents = []
            for fact in custom_facts:
                doc = Document(
                    page_content=fact["content"],
                    metadata=fact["metadata"]
                )
                documents.append(doc)

            if documents:
                self.db.add_documents(documents)
                self.db.persist()
                logger.info(f"✅ {len(documents)} Fakten zur Datenbank hinzugefügt")

        except Exception as e:
            logger.error(f"Fehler beim Hinzufügen von Fakten: {e}")

    def _initialize_vector_store(self):
        """Initialisiert die Vektor-Datenbank"""
        if os.path.exists(self.config.db_dir) and len(os.listdir(self.config.db_dir)) > 0:
            self.db = Chroma(persist_directory=self.config.db_dir, embedding_function=self.emb)
        else:
            self.db = Chroma(persist_directory=self.config.db_dir, embedding_function=self.emb)
            self._ensure_index()

    def _ensure_index(self):
        """Indexiert Dokumente"""
        if not os.path.exists(self.config.data_folder):
            os.makedirs(self.config.data_folder)
            return

        try:
            loader = DirectoryLoader(self.config.data_folder)
            docs = loader.load()
            if docs:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                split_docs = text_splitter.split_documents(docs)
                self.db.add_documents(split_docs)
                self.db.persist()
        except Exception as e:
            logger.error(f"Indexierung fehlgeschlagen: {e}")

    def _show_capabilities(self):
        """Zeigt Fähigkeiten"""
        print("\n" + "=" * 60)
        print("🤖 Intelligente KI - Fähigkeiten")
        print("=" * 60)
        print("🌐 Internet-Zugriff:")
        print("   - Web-Suche (DuckDuckGo, Wikipedia)")
        print("   - Web-Scraping")
        print("   - Echtzeit-Daten")

        print("\n🔬 Test-Umgebung:")
        print("   - Python Code Ausführung")
        print("   - Code Testing")

        print("\n🧠 Erweiterte Analyse:")
        print("   - Multi-Step Reasoning")
        print("   - Komplexe Problemanalyse")

        print("\n💡 Beispiel-Fragen:")
        print("   'Analysiere aktuelle KI-News'")
        print("   'Schreibe einen Python Web-Scraper'")
        print("   'Erkläre Machine Learning'")
        print("=" * 60)

    def _create_qa_chain(self):
        """Erstellt QA Chain - KORRIGIERTE VERSION"""
        # Vereinfachtes Prompt ohne internet_data Parameter
        prompt = PromptTemplate(
            template="""Du bist eine hilfreiche KI.

Kontext: {context}

Frage: {question}

Antworte strukturiert und hilfreich. Wenn du die Antwort nicht weißt, sage das ehrlich.

Antwort:""",
            input_variables=["context", "question"]  # Nur diese beiden Parameter
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.db.as_retriever(search_kwargs={"k": self.config.k}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def ask(self, question: str) -> str:
        """Fragebeantwortung mit erweiterten Fähigkeiten - KORRIGIERT"""
        try:
            logger.info(f"Verarbeite Frage: {question}")

            # Entscheide welche Fähigkeiten benötigt werden
            capabilities_needed = self._analyze_capability_needs(question)

            # Verwende erweiterte Analyse für komplexe Fragen
            if 'analysis' in capabilities_needed and self.advanced_analysis:
                return self.advanced_analysis.analyze_problem(question)

            # Für Internet-Fragen: separate Verarbeitung
            if 'internet' in capabilities_needed and self.internet_search:
                return self._handle_internet_question(question)

            # Normale Verarbeitung ohne internet_data Parameter
            response = self.qa_chain.invoke({"query": question})
            answer = response.get("result", "Ich konnte keine Antwort finden.")

            # Füge Code-Ausführung hinzu falls benötigt
            if 'code' in capabilities_needed and self.test_env:
                answer += self._add_code_execution(question, answer)

            return self._format_answer(answer, capabilities_needed)

        except Exception as e:
            logger.error(f"KI Fehler: {e}")
            return f"Entschuldigung, ein Fehler ist aufgetreten: {e}"

    def _handle_internet_question(self, question: str) -> str:
        """Behandelt Internet-Fragen separat"""
        try:
            # Sammle Internet-Daten
            internet_data = self._gather_internet_data(question)

            # Erstelle eine Antwort mit den Internet-Daten
            response_prompt = f"""
Frage: {question}

Gefundene Informationen:
{internet_data}

Beantworte die Frage basierend auf den oben genannten Informationen.
Sei präzise und gib die Quellen an.
"""

            answer = self.llm.invoke(response_prompt)
            return self._format_answer(answer, ['internet'])

        except Exception as e:
            return f"Internet-Recherche fehlgeschlagen: {e}"

    def _analyze_capability_needs(self, question: str) -> List[str]:
        """Analysiert welche Fähigkeiten benötigt werden"""
        capabilities = []
        question_lower = question.lower()

        if any(word in question_lower for word in ['internet', 'web', 'online', 'aktuell', 'news', 'suche']):
            capabilities.append('internet')

        if any(word in question_lower for word in ['code', 'programm', 'test', 'ausführen', 'python', 'script']):
            capabilities.append('code')

        if any(word in question_lower for word in ['komplex', 'problem', 'analysiere', 'löse', 'warum', 'erkläre']):
            capabilities.append('analysis')

        return capabilities if capabilities else ['basic']

    def _gather_internet_data(self, question: str) -> str:
        """Sammelt Internet-Daten"""
        internet_info = "🌐 Internet-Recherche:\n\n"

        try:
            search_results = self.internet_search.web_search(question, 3)
            for result in search_results:
                internet_info += f"🔍 {result['title']}\n"
                internet_info += f"   {result['content'][:150]}...\n"
                internet_info += f"   Quelle: {result['source']}\n\n"

            if not search_results:
                internet_info += "Keine relevanten Informationen gefunden.\n"

        except Exception as e:
            internet_info += f"Internet-Recherche fehlgeschlagen: {e}\n"

        return internet_info

    def _add_code_execution(self, question: str, current_answer: str) -> str:
        """Fügt Code-Ausführung hinzu"""
        try:
            code_prompt = f"""
Erstelle Python-Code für:

Frage: {question}
Kontext: {current_answer[:400]}

Generiere ausführbaren Code.
Antworte nur mit Python-Code.
"""

            code = self.llm.invoke(code_prompt)

            # Führe Code aus
            execution_result = self.test_env.execute_python_code(code)

            code_section = "\n\n💻 Code-Ausführung:\n```python\n"
            code_section += code + "\n```\n"

            if execution_result['success']:
                code_section += f"✅ Output:\n{execution_result['stdout']}\n"
            else:
                code_section += f"❌ Fehler:\n{execution_result['stderr']}\n"

            return code_section

        except Exception as e:
            return f"\n\nCode-Ausführung fehlgeschlagen: {e}"

    def _format_answer(self, answer: str, capabilities: List[str]) -> str:
        """Formatiert die Antwort"""
        formatted = "\n" + "=" * 50 + "\n"
        formatted += "🤖 KI Antwort\n"
        formatted += "=" * 50 + "\n\n"

        formatted += answer

        if len(capabilities) > 1 or 'basic' not in capabilities:
            formatted += f"\n\nVerwendete Fähigkeiten: {', '.join(capabilities)}"

        formatted += "\n" + "=" * 50
        return formatted

    def create_web_scraper(self, url: str) -> str:
        """Erstellt einen Web-Scraper"""
        if not self.internet_search:
            return "Internet-Zugriff nicht aktiviert"

        try:
            scraper_code = f'''
import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    """Web-Scraper"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title')
        title_text = title.text.strip() if title else "No title"

        paragraphs = soup.find_all('p')
        content = ' '.join([p.text.strip() for p in paragraphs[:3] if p.text.strip()])

        return {{
            'success': True,
            'title': title_text,
            'content': content[:300],
            'url': url
        }}
    except Exception as e:
        return {{'success': False, 'error': str(e)}}

# Ausführung
result = scrape_website("{url}")
print("=== Web Scraper Results ===")
print(f"URL: {{result.get('url')}}")
print(f"Titel: {{result.get('title')}}")
print(f"Inhalt: {{result.get('content')}}")
print(f"Erfolg: {{result.get('success')}}")
'''

            # Teste den Scraper
            test_result = self.test_env.execute_python_code(scraper_code)

            result_text = f"🌐 Web-Scraper für: {url}\n\n"
            result_text += "```python\n" + scraper_code + "\n```\n\n"

            if test_result['success']:
                result_text += f"✅ Test erfolgreich:\n{test_result['stdout']}"
            else:
                result_text += f"⚠️ Test mit Fehlern:\n{test_result['stderr']}"

            return result_text

        except Exception as e:
            return f"Scraper-Erstellung fehlgeschlagen: {e}"


# -----------------------------
# Hauptfunktion
# -----------------------------
def main():
    config = AIConfig()

    try:
        print("🚀 Starte KI...")
        ai = IntelligentAI(config)

        while True:
            print("\n" + "=" * 50)
            user_input = input("Frage: ").strip()

            if user_input.lower() == 'exit':
                print("KI beendet!")
                break
            elif user_input.lower() == 'scraper':
                url = input("URL: ").strip()
                result = ai.create_web_scraper(url)
                print(result)
            elif user_input.lower().startswith('http'):
                result = ai.create_web_scraper(user_input)
                print(result)
            elif user_input:
                print("KI denkt nach...")
                result = ai.ask(user_input)
                print(result)
            else:
                print("Bitte Frage eingeben!")

    except KeyboardInterrupt:
        print("\nKI beendet!")
    except Exception as e:
        print(f"Fehler: {e}")


if __name__ == "__main__":
    main()