import signal
import time
from typing import Dict, Union

import openai
import tiktoken

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI()
    return _client


def num_tokens_from_messages(message, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if isinstance(message, list):
        # use last message.
        num_tokens = len(encoding.encode(message[0]["content"]))
    else:
        num_tokens = len(encoding.encode(message))
    return num_tokens


def create_chatgpt_config(
    message: Union[str, list],
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "gpt-3.5-turbo",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [{"role": "system", "content": system_message}] + message,
        }
    else:
        config = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "n": batch_size,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": message},
            ],
        }
    return config


def handler(signum, frame):
    # swallow signum and frame
    raise Exception("end of time")


# signal.alarm / SIGALRM are Unix-only; use a no-op on Windows
_HAS_SIGALRM = hasattr(signal, "SIGALRM")

def _set_alarm(seconds):
    if _HAS_SIGALRM:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(seconds)

def _cancel_alarm():
    if _HAS_SIGALRM:
        signal.alarm(0)


def request_chatgpt_engine(config):
    ret = None
    while ret is None:
        try:
            _set_alarm(100)
            ret = _get_client().chat.completions.create(**config)
            _cancel_alarm()
        except openai._exceptions.BadRequestError as e:
            print(e)
            _cancel_alarm()
        except openai._exceptions.RateLimitError as e:
            print("Rate limit exceeded. Waiting...")
            print(e)
            _cancel_alarm()
            time.sleep(5)
        except openai._exceptions.APIConnectionError as e:
            print("API connection error. Waiting...")
            _cancel_alarm()
            time.sleep(5)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            _cancel_alarm()
            time.sleep(1)
    return ret


def create_anthropic_config(
    message: str,
    prefill_message: str,
    max_tokens: int,
    temperature: float = 1,
    batch_size: int = 1,
    system_message: str = "You are a helpful assistant.",
    model: str = "claude-2.1",
) -> Dict:
    if isinstance(message, list):
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": message,
        }
    else:
        config = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "system": system_message,
            "messages": [
                {"role": "user", "content": message},
                {"role": "assistant", "content": prefill_message},
            ],
        }
    return config


def request_anthropic_engine(client, config):
    ret = None
    while ret is None:
        try:
            _set_alarm(100)
            ret = _get_client().messages.create(**config)
            _cancel_alarm()
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            _cancel_alarm()
            time.sleep(10)
    return ret
