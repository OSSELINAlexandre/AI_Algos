from langchain_ollama import OllamaLLM
from bs4 import BeautifulSoup
from ddgs import DDGS
import requests
import time

def search_and_scrape(query, max_results=3):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    scraped_contents = []

    print(f"[🔎] Recherche de résultats pour : {query}")
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)

        for result in results:
            url = result.get("href")
            if not url:
                continue

            try:
                print(f"[📄] Scraping de : {url}")
                res = requests.get(url, headers=headers, timeout=10)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, 'html.parser')
                for tag in soup(["script", "style"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                scraped_contents.append(text[:3000])  # Limiter à 3000 caractères par page
                time.sleep(1)  # Pour éviter le blocage
            except Exception as e:
                print(f"[⚠️] Erreur pour {url}: {e}")
                continue

    return scraped_contents


def synthesize_with_llm(texts, query):
    llm = OllamaLLM(model="llama3:latest", temperature=0.7)

    joined_text = "\n\n---\n\n".join(texts)
    prompt = f"""
Tu es un assistant de recherche intelligent.

Voici la question utilisateur :
"{query}"

Voici des extraits de pages web :
{joined_text}

En te basant uniquement sur ces informations, donne une réponse claire et synthétique à l'utilisateur.
Si tu n'as pas assez d'informations, indique-le explicitement.
"""

    return llm.invoke(prompt)


def main():
    print("=== Agent IA Recherche & Synthèse ===")
    print("Tape 'quit' pour quitter.\n")

    while True:
        question = input("❓ Question : ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Au revoir !")
            break

        scraped_data = search_and_scrape(question)
        if not scraped_data:
            print("❌ Aucun contenu utile trouvé.")
            continue

        print("\n[🧠] Synthèse en cours...")
        final_answer = synthesize_with_llm(scraped_data, question)
        print("\n✅ Réponse :")
        print("-" * 40)
        print(final_answer)
        print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
