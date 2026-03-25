import os
import json
import uuid
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Generator
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

try:
    from tavily import TavilyClient
except ImportError:
    raise ImportError("Please install tavily-python: uv add tavily")

load_dotenv()

DASHSCOPE_MODELS = {
    "qwen-plus": "qwen-plus",
    "qwen-max": "qwen-max",
    # "qwen-turbo": "qwen-turbo",
    # "qwen2-72b-instruct": "qwen2-72b-instruct",
    # "qwen2.5-72b-instruct": "qwen2.5-72b-instruct",
    # "qwen2.5-32b-instruct": "qwen2.5-32b-instruct",
    # "qwen2.5-7b-instruct": "qwen2.5-7b-instruct",
    # "qwen2.5-14b-instruct": "qwen2.5-14b-instruct",
    # "qwen2.5-1.5b-instruct": "qwen2.5-1.5b-instruct",
}

class DashscopeTavilyInference:
    def __init__(
        self,
        model_name: str = "qwen-plus",
        base_url: str | None = None,
        api_key: str | None = None,
        tavily_api_key: str | None = None,
        log_dir: str = "./logs"
    ):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError("DASHSCOPE_API_KEY not set. Please set DASHSCOPE_API_KEY or pass it as argument.")

        self.base_url = base_url or os.environ.get("DASHSCOPE_BASE_URL")
        if not self.base_url:
            raise ValueError("DASHSCOPE_BASE_URL not set. Please set DASHSCOPE_BASE_URL or pass it as argument.")

        if model_name not in DASHSCOPE_MODELS:
            raise ValueError(f"Invalid model: {model_name}. Available: {', '.join(DASHSCOPE_MODELS.keys())}")
        self.model_name = DASHSCOPE_MODELS[model_name]

        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("TAVILY_API_KEY not set. Please set TAVILY_API_KEY or pass it as argument.")
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = 5
        self.initial_backoff = 1
        self.max_backoff = 60
        self.backoff_factor = 2
        self.jitter = 0.1

    def _backoff_time(self, retry: int) -> float:
        backoff = min(self.max_backoff, self.initial_backoff * (self.backoff_factor ** retry))
        jitter_amount = backoff * self.jitter
        backoff = backoff + random.uniform(-jitter_amount, jitter_amount)
        return max(0, backoff)

    def _log_to_jsonl(self, log_entry: Dict[str, Any]) -> str:
        log_id = log_entry.get("log_id", str(uuid.uuid4()))
        log_entry["log_id"] = log_id
        log_file = self.log_dir / f"dashscope_tavily_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        return log_id

    def _search_with_tavily(self, query: str) -> Dict[str, Any]:
        log_entry = {
            "log_id": str(uuid.uuid4()),
            "type": "tavily_search",
            "timestamp": datetime.now().isoformat(),
            "input": {"query": query},
            "output": None,
            "error": None
        }
        try:
            start = time.time()
            result = self.tavily_client.search(
                query=query,
                max_results=5,
                include_answer=True,
                include_raw_content=False
            )
            duration = time.time() - start
            log_entry["output"] = result
            log_entry["duration_seconds"] = duration
            self._log_to_jsonl(log_entry)
            return result
        except Exception as e:
            log_entry["error"] = str(e)
            self._log_to_jsonl(log_entry)
            raise

    def generate_response(
        self,
        input_text: str,
        use_tools: bool = False,
        stream: bool = False
    ) -> Union[str, Dict]:
        messages = [{"role": "user", "content": input_text}]
        session_id = str(uuid.uuid4())

        if not use_tools:
            return self._plain_chat(input_text, messages, session_id)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "tavily_search",
                    "description": "Search the web for current information about any topic. Use this when you need up-to-date information that may not be in your training data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find information about the topic."
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]

        tool_call_log = []

        for retry in range(self.max_retries):
            try:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)

                llm_log_entry = {
                    "log_id": str(uuid.uuid4()),
                    "type": "dashscope_llm",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "input": {
                        "model": self.model_name,
                        "messages": messages,
                        "tools": tools,
                        "use_tools": True
                    },
                    "output": None,
                    "error": None,
                    "tool_calls": []
                }

                start = time.time()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=1,
                    max_tokens=4096
                )
                duration = time.time() - start
                llm_log_entry["duration_seconds"] = duration

                choice = response.choices[0]
                assistant_message = choice.message
                llm_log_entry["output"] = {
                    "content": assistant_message.content,
                    "tool_calls": [
                        {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                         } for tc in (assistant_message.tool_calls or [])
                    ]
                }

                if assistant_message.tool_calls:
                    messages.append(assistant_message.model_dump(exclude_none=True))

                    for tc in assistant_message.tool_calls:
                        if tc.function.name == "tavily_search":
                            args = json.loads(tc.function.arguments)
                            query = args["query"]

                            search_result = self._search_with_tavily(query)
                            tool_call_log.append({
                                "tool": "tavily_search",
                                "query": query,
                                "result": search_result
                            })

                            search_summary = f"Search results for '{query}':\n"
                            for r in search_result.get("results", [])[:3]:
                                search_summary += f"- {r.get('title', 'N/A')}: {r.get('url', 'N/A')}\n"
                            if search_result.get("answer"):
                                search_summary += f"\nAnswer: {search_result['answer']}"

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": search_summary
                            })

                    second_response = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=1,
                        max_tokens=4096
                    )

                    second_log_entry = {
                        "log_id": str(uuid.uuid4()),
                        "type": "dashscope_llm",
                        "timestamp": datetime.now().isoformat(),
                        "session_id": session_id,
                        "input": {"model": self.model_name, "messages": messages},
                        "output": {
                            "content": second_response.choices[0].message.content,
                            "tool_calls": []
                        },
                        "error": None,
                        "duration_seconds": time.time() - start
                    }
                    llm_log_entry["tool_calls"] = tool_call_log
                    self._log_to_jsonl(llm_log_entry)
                    self._log_to_jsonl(second_log_entry)

                    return second_response.choices[0].message.content or ""

                llm_log_entry["tool_calls"] = []
                self._log_to_jsonl(llm_log_entry)
                return assistant_message.content or ""

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if retry < self.max_retries - 1:
                        backoff_time = self._backoff_time(retry)
                        print(f"Rate limit. Retrying in {backoff_time:.2f}s (attempt {retry+1}/{self.max_retries})...")
                        time.sleep(backoff_time)
                    else:
                        return f"Error: Rate limit exceeded after {self.max_retries} retries."
                else:
                    return f"Error: {str(e)}"

        return "Error: Maximum retries exceeded."

    def _plain_chat(self, input_text: str, messages: List[Dict], session_id: str) -> str:
        for retry in range(self.max_retries):
            try:
                client = OpenAI(api_key=self.api_key, base_url=self.base_url)

                llm_log_entry = {
                    "log_id": str(uuid.uuid4()),
                    "type": "dashscope_llm",
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id,
                    "input": {
                        "model": self.model_name,
                        "messages": messages,
                        "use_tools": False
                    },
                    "output": None,
                    "error": None,
                    "duration_seconds": None
                }

                start = time.time()
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=1,
                    max_tokens=4096
                )
                duration = time.time() - start
                llm_log_entry["duration_seconds"] = duration

                content = response.choices[0].message.content
                llm_log_entry["output"] = {"content": content, "tool_calls": []}
                self._log_to_jsonl(llm_log_entry)
                return content or ""

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "rate limit" in error_str:
                    if retry < self.max_retries - 1:
                        backoff_time = self._backoff_time(retry)
                        print(f"Rate limit. Retrying in {backoff_time:.2f}s (attempt {retry+1}/{self.max_retries})...")
                        time.sleep(backoff_time)
                    else:
                        return f"Error: Rate limit exceeded after {self.max_retries} retries."
                else:
                    return f"Error: {str(e)}"

        return "Error: Maximum retries exceeded."


def run_inference_multithread(
    model_name: str,
    input_list: List[str],
    use_tools: bool = True,
    max_workers: int = 4,
    base_url: str | None = None,
    api_key: str | None = None,
    tavily_api_key: str | None = None,
    log_dir: str = "./logs"
) -> List[Union[str, Dict]]:
    inference = DashscopeTavilyInference(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        tavily_api_key=tavily_api_key,
        log_dir=log_dir
    )

    def process_input(input_text: str) -> Union[str, Dict]:
        return inference.generate_response(input_text, use_tools=use_tools)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_input, text) for text in input_list]
        for f in tqdm(futures, total=len(futures), desc="Model inference"):
            results.append(f.result())

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dashscope_tavily_inference.py <input_text> [--use-tools]")
        sys.exit(1)

    input_text = sys.argv[1]
    use_tools = "--use-tools" in sys.argv

    print(f"Input: {input_text}")
    print(f"Use tools: {use_tools}\n")

    inference = DashscopeTavilyInference()

    if use_tools:
        print("With tool calling (Tavily search):")
        result = inference.generate_response(input_text, use_tools=True)
    else:
        print("Without tool calling:")
        result = inference.generate_response(input_text, use_tools=False)

    print(f"\nResult: {result}")
    print(f"\nLogs saved to: {inference.log_dir}")
