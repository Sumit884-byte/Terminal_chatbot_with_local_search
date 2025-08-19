import ollama
import trafilatura
import sys_msgs
import requests
from bs4 import BeautifulSoup
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from collections import deque
import re
import sys
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import asyncio

# --- NLTK Setup for Lemmatization ---
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")

# --- Config ---
MODEL = "llama3.2:1b"
YEAR = datetime.now().year
IDLE_WAIT = 5  # seconds

conversation_history = [
    {"role": "assistant", "content": sys_msgs.assistant_msg}
]

# Queues
queue_store = {"tasks": []}
deep_queue_store = deque()

# Track idle + background output
last_output_time = time.time()
idle_results = deque()


# --- Utility ---
async def safe_messages(messages):
    """Ensure all messages in the list are valid dictionaries."""
    return [m for m in messages if isinstance(m, dict) and "role" in m and "content" in m]


lemmatizer = WordNetLemmatizer()


async def lemmatize_text(text):
    """Lemmatizes the input text to find the root form of words."""
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(lemmatized_tokens)


# --- Auto Search Keyword Detection ---
SEARCH_KEYWORDS = [
    "latest", "hackathon", "conference", "release", "update",
    "event", "news", "documentation", "changelog", "api",
    "tutorial", "guide", "new", "2025", "2026"
]


async def should_auto_search(prompt):
    """Check if the prompt contains keywords that trigger an automatic search."""
    text = prompt.lower()
    return any(keyword in text for keyword in SEARCH_KEYWORDS)


async def get_plan_from_ai(prompt):
    """
    Determines the execution plan based on the user's prompt.
    - Detects error-related tasks.
    - Handles forced searches (starting with '/').
    - Triggers automatic searches for certain keywords.
    - Defaults to a direct answer.
    """
    if await detect_error_task(prompt):  # error detected
        return "ERROR", prompt
    elif prompt.startswith("/"):  # forced search
        return "SEARCH", prompt[1:].strip()
    elif await should_auto_search(prompt):  # auto search
        return "SEARCH", prompt
    else:
        return "ANSWER", prompt


# --- Search + Scraping ---
async def duckduckgo(query):
    """Performs a search using DuckDuckGo and returns formatted results."""
    headers = {"User-Agent": "Mozilla/5.0"}
    url = f'https://html.duckduckgo.com/html/?q={query}'
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for i, result in enumerate(soup.find_all('div', class_='result')):
            title_tag = result.find('a', class_='result__a')
            snippet_tag = result.find('a', class_='result__snippet')
            if not title_tag or not snippet_tag:
                continue
            results.append({
                'id': i,
                'link': title_tag['href'],
                'title': title_tag.text,
                'snippet': snippet_tag.text.strip(),
            })
        return results
    except requests.RequestException as e:
        print(f"Error during search: {e}")
        return []


async def scrape_webpage(url, timeout=12):
    """Fetches and extracts clean text from a webpage using trafilatura."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            extracted = trafilatura.extract(downloaded)
            if extracted:
                return extracted.strip()
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text.strip()
    except Exception:
        return ""


async def scrape_all_results(queries, batch_size=10, min_words=200, hard_limit=10):
    """
    Scrapes search results in batches and merges the text content.
    It continues scraping until a minimum word count is met or a hard limit of pages is reached.
    """
    all_results = []
    for query in queries:
        results = await duckduckgo(query)
        if results:
            all_results.extend(results)

    if not all_results:
        return ""

    merged_texts = []
    seen_links = set()
    start = 0

    while start < len(all_results) and start < hard_limit:
        subset = all_results[start:start + batch_size]
        for res in subset:
            link = res.get("link")
            if link in seen_links:
                continue
            seen_links.add(link)

            page_text = await scrape_webpage(link)
            if page_text:
                merged_texts.append(page_text)
            else:
                merged_texts.append(res['title'] + ". " + res['snippet'])

        start += batch_size

        merged = "\n\n".join(merged_texts)
        if len(merged.split()) >= min_words:
            return merged

    return "\n\n".join(merged_texts)


# --- Cleanup ---
async def cleanup_response_with_ai(raw_text, entity=None):
    """Uses the AI to clean up and refine a raw text response."""
    try:
        messages = [
            {"role": "system", "content": (
                "You are a cleanup assistant. "
                "Remove irrelevant or repeated information, disclaimers, and produce a clean answer. "
                f"The current year is {YEAR}. "
                "Stay strictly on the given entity/topic."
            )},
            {"role": "user", "content": f"Entity: {entity or 'Unknown'}\nRaw answer: {raw_text}"}
        ]
        response = ollama.chat(
            model=MODEL,
            messages=messages,
            options={"temperature": 0}
        )
        return response['message']['content'].strip()
    except Exception:
        return raw_text.strip()


# --- Error Detection ---
async def detect_error_task(prompt):
    """Detects if a prompt is describing a software error."""
    error_keywords = ["Error", "Exception", "Traceback", "AttributeError",
                      "ImportError", "ModuleNotFoundError", "TypeError", "ValueError"]
    return any(word in prompt for word in error_keywords)


# --- AI Response ---
async def stream_assistant_response(entity=None, background=False, task_name=None):
    """
    Generates and streams the AI's response.
    - For foreground tasks, it prints the response token by token.
    - For background tasks, it generates the full response, cleans it, and then displays it.
    """
    global conversation_history, last_output_time

    try:
        messages = await safe_messages(conversation_history)
        response_stream = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options={"temperature": 0}
        )
        complete_response = ''

        if not background:
            print("\nASSISTANT:")

        for chunk in response_stream:
            if 'content' in chunk['message']:
                content_piece = chunk['message']['content']
                complete_response += content_piece
                if not background:
                    sys.stdout.write(content_piece)
                    sys.stdout.flush()

        if not background:
            print()
            conversation_history.append({'role': 'assistant', 'content': complete_response.strip()})
            last_output_time = time.time()
        else:
            cleaned = await cleanup_response_with_ai(complete_response, entity=entity)
            conversation_history.append({'role': 'assistant', 'content': cleaned})

            if time.time() - last_output_time > IDLE_WAIT:
                print(f"\n\n============================================================")
                print(f"🕒 Background result (task: {task_name})")
                print("============================================================")
                print("ASSISTANT:")
                print(cleaned + "\n")
                last_output_time = time.time()
            else:
                idle_results.append((task_name, cleaned))

    except Exception as e:
        error_msg = f'⚠️ Error during AI response generation: {str(e)}'
        conversation_history.append({'role': 'assistant', 'content': error_msg})
        if not background:
            print(f"\nASSISTANT:\n{error_msg}\n")


# --- Task Processor ---
async def process_task(task, deep=False):
    """Processes a single task based on the determined action plan."""
    action, data = await get_plan_from_ai(task)

    if action == "SEARCH":
        lemmatized_query = await lemmatize_text(data)
        print(f"🔎 Lemmatized Search: {lemmatized_query}")
        merged_context = await scrape_all_results([lemmatized_query], min_words=500)
        if merged_context.strip():
            prompt_with_context = (
                f"The current year is {YEAR}.\n\n"
                f"Using ONLY the following merged search results, "
                f"give a single, clear, concise answer.\n\n"
                f"SEARCH RESULTS:\n---\n{merged_context[:12000]}\n---\n\n"
                f"USER QUESTION: {task}"
            )
            conversation_history.append({'role': 'user', 'content': prompt_with_context})
            await stream_assistant_response(entity=data, background=deep, task_name=task)
        else:
            print("⚠️ No search context found — skipping response.")

    elif action == "ANSWER":
        conversation_history.append({'role': 'user', 'content': data})
        await stream_assistant_response(entity=data, background=deep, task_name=task)


# --- Background Worker ---
async def deepqueue_worker():
    """A worker thread that continuously processes tasks from the deep queue."""
    while True:
        if deep_queue_store:
            task = deep_queue_store.popleft()
            await process_task(task, deep=True)
        else:
            await asyncio.sleep(2)


# --- Queue Management ---
async def add_multiple_to_queue_interactive(deep=False):
    """Allows the user to interactively add multiple tasks to a queue."""
    global queue_store, deep_queue_store
    print("👉 Enter tasks one by one. Press Enter on an empty line to finish.")
    while True:
        task = input("TASK: ")
        if not task.strip():
            break
        if deep:
            deep_queue_store.append(task)
            print(f"✅ Queued task (deep): {task}")
        else:
            queue_store["tasks"].append(task)
            print(f"✅ Queued task: {task} (deep=False)")


# --- Main Loop ---
async def main():
    """The main function to run the chat application."""
    global last_output_time
    asyncio.create_task(deepqueue_worker())

    print(f"ASSISTANT: {sys_msgs.assistant_msg}")

    while True:
        if idle_results and (time.time() - last_output_time > IDLE_WAIT):
            task_name, cleaned = idle_results.popleft()
            print(f"\n\n============================================================")
            print(f"🕒 Background result (task: {task_name})")
            print("============================================================")
            print("ASSISTANT:")
            print(cleaned + "\n")
            last_output_time = time.time()

        prompt = input("USER: \n")
        if prompt.lower() in ['exit', 'quit']:
            break

        if prompt.startswith("/queue"):
            await add_multiple_to_queue_interactive(deep=False)
            while queue_store["tasks"]:
                task = queue_store["tasks"].pop(0)
                print("\n============================================================")
                print(f"⚡ Processing queued task: {task}")
                print("============================================================")
                await process_task(task, deep=False)

        elif prompt.startswith("/deepqueue"):
            await add_multiple_to_queue_interactive(deep=True)

        else:
            await process_task(prompt, deep=False)


if __name__ == "__main__":
    asyncio.run(main())
