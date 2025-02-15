#!/usr/bin/env python3
"""
utils.py - Utility functions and definitions for the project.

This file contains helper functions for:
    - Loading and configuring language models (LLMs) from various providers.
    - Handling API keys and errors.
    - File and image processing (e.g. encoding images, retrieving latest files).
    - Other miscellaneous utilities.

All functions and constants are documented below. Ensure this file is updated and saved 
to run your project locally.
"""

# ==============================================================
# IMPORTS
# ==============================================================

import base64
import os
import time
from pathlib import Path
from typing import Dict, Optional
import requests

# Import LLM client libraries
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import gradio as gr

# Import custom LLM classes for DeepSeek for Ollama
from .llm import DeepSeekR1ChatOpenAI, DeepSeekR1ChatOllama

# ==============================================================
# GLOBAL CONSTANTS & DICTIONARIES
# ==============================================================

# Display names for providers (used in error messages and UI)
PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google"
}

# Predefined model names for each provider.
# Updated the "ollama" list to include "deepseek-r1:1.5b" along with other DeepSeek models.
model_names = {
    "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"],
    "openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o3-mini"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "google": [
        "gemini-2.0-flash-exp", 
        "gemini-2.0-flash-thinking-exp", 
        "gemini-1.5-flash-latest", 
        "gemini-1.5-flash-8b-latest", 
        "gemini-2.0-flash-thinking-exp-01-21"
    ],
    "ollama": [
        "qwen2.5:7b", 
        "llama2:7b", 
        "deepseek-r1:1.5b",  # New model added
        "deepseek-r1:14b", 
        "deepseek-r1:32b"
    ],
    "azure_openai": ["gpt-4o", "gpt-4", "gpt-3.5-turbo"],
    "mistral": ["pixtral-large-latest", "mistral-large-latest", "mistral-small-latest", "ministral-8b-latest"]
}

# ==============================================================
# FUNCTIONS (Outside Classes)
# ==============================================================

def get_llm_model(provider: str, **kwargs):
    """
    Retrieve an LLM model instance based on the provider and configuration parameters.

    For non-Ollama providers, an API key is required (either from kwargs or environment variable).
    For Ollama, no API key is needed; instead, a local endpoint is used.

    Parameters:
        provider (str): The LLM provider (e.g., "openai", "ollama", etc.).
        **kwargs: Additional configuration parameters such as model_name, temperature, base_url, and api_key.

    Returns:
        An instance of the selected language model client.
    """
    # For providers other than Ollama, require an API key.
    if provider not in ["ollama"]:
        env_var = f"{provider.upper()}_API_KEY"
        api_key = kwargs.get("api_key", "") or os.getenv(env_var, "")
        if not api_key:
            handle_api_key_error(provider, env_var)
        kwargs["api_key"] = api_key

    if provider == "anthropic":
        base_url = kwargs.get("base_url", "https://api.anthropic.com")
        return ChatAnthropic(
            model_name=kwargs.get("model_name", "claude-3-5-sonnet-20240620"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )
    elif provider == "mistral":
        base_url = kwargs.get("base_url", os.getenv("MISTRAL_ENDPOINT", "https://api.mistral.ai/v1"))
        api_key = kwargs.get("api_key", os.getenv("MISTRAL_API_KEY", ""))
        return ChatMistralAI(
            model=kwargs.get("model_name", "mistral-large-latest"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=api_key,
        )
    elif provider == "openai":
        base_url = kwargs.get("base_url", os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1"))
        return ChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            base_url=base_url,
            api_key=kwargs["api_key"],
        )
    elif provider == "deepseek":
        base_url = kwargs.get("base_url", os.getenv("DEEPSEEK_ENDPOINT", ""))
        if kwargs.get("model_name", "deepseek-chat") == "deepseek-reasoner":
            return DeepSeekR1ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-reasoner"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=kwargs["api_key"],
            )
        else:
            return ChatOpenAI(
                model=kwargs.get("model_name", "deepseek-chat"),
                temperature=kwargs.get("temperature", 0.0),
                base_url=base_url,
                api_key=kwargs["api_key"],
            )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=kwargs.get("model_name", "gemini-2.0-flash-exp"),
            temperature=kwargs.get("temperature", 0.0),
            google_api_key=kwargs["api_key"],
        )
    elif provider == "ollama":
        base_url = kwargs.get("base_url", os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"))
        # Check if the model name contains "deepseek-r1". If so, use the DeepSeekR1ChatOllama class.
        if "deepseek-r1" in kwargs.get("model_name", "qwen2.5:7b"):
            return DeepSeekR1ChatOllama(
                model=kwargs.get("model_name", "deepseek-r1:14b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                base_url=base_url,
            )
        else:
            return ChatOllama(
                model=kwargs.get("model_name", "qwen2.5:7b"),
                temperature=kwargs.get("temperature", 0.0),
                num_ctx=kwargs.get("num_ctx", 32000),
                num_predict=kwargs.get("num_predict", 1024),
                base_url=base_url,
            )
    elif provider == "azure_openai":
        base_url = kwargs.get("base_url", os.getenv("AZURE_OPENAI_ENDPOINT", ""))
        return AzureChatOpenAI(
            model=kwargs.get("model_name", "gpt-4o"),
            temperature=kwargs.get("temperature", 0.0),
            api_version="2024-05-01-preview",
            azure_endpoint=base_url,
            api_key=kwargs["api_key"],
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def update_model_dropdown(llm_provider, api_key=None, base_url=None):
    """
    Update the model name dropdown with predefined models for the selected provider.

    Parameters:
        llm_provider (str): Selected provider key.
        api_key (str): API key (optional).
        base_url (str): Base URL for the API endpoint (optional).

    Returns:
        A Gradio Dropdown component with the available model choices.
    """
    if not api_key:
        api_key = os.getenv(f"{llm_provider.upper()}_API_KEY", "")
    if not base_url:
        base_url = os.getenv(f"{llm_provider.upper()}_BASE_URL", "")
    if llm_provider in model_names:
        return gr.Dropdown(choices=model_names[llm_provider], value=model_names[llm_provider][0], interactive=True)
    else:
        return gr.Dropdown(choices=[], value="", interactive=True, allow_custom_value=True)

def handle_api_key_error(provider: str, env_var: str):
    """
    Raises a Gradio Error indicating that the API key for the provider is missing.

    Parameters:
        provider (str): The provider name.
        env_var (str): The expected environment variable name for the API key.

    Raises:
        gr.Error: With a message explaining which API key is missing.
    """
    provider_display = PROVIDER_DISPLAY_NAMES.get(provider, provider.upper())
    raise gr.Error(
        f"💥 {provider_display} API key not found! 🔑 Please set the `{env_var}` environment variable or provide it in the UI."
    )

def encode_image(img_path):
    """
    Encodes an image at the given path into a base64 string.

    Parameters:
        img_path (str): The path to the image file.

    Returns:
        A base64 encoded string of the image data, or None if the image path is invalid.
    """
    if not img_path:
        return None
    with open(img_path, "rb") as fin:
        image_data = base64.b64encode(fin.read()).decode("utf-8")
    return image_data

def get_latest_files(directory: str, file_types: list = ['.webm', '.zip']) -> Dict[str, Optional[str]]:
    """
    Retrieves the latest file for each specified file type in the directory.

    Parameters:
        directory (str): The directory to search.
        file_types (list): List of file extensions to search for.

    Returns:
        A dictionary mapping each file extension to the latest file path (or None if not found).
    """
    latest_files: Dict[str, Optional[str]] = {ext: None for ext in file_types}
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        return latest_files
    for file_type in file_types:
        try:
            matches = list(Path(directory).rglob(f"*{file_type}"))
            if matches:
                latest = max(matches, key=lambda p: p.stat().st_mtime)
                # Only return files that are complete (not currently being written)
                if time.time() - latest.stat().st_mtime > 1.0:
                    latest_files[file_type] = str(latest)
        except Exception as e:
            print(f"Error getting latest {file_type} file: {e}")
    return latest_files

async def capture_screenshot(browser_context):
    """
    Captures a screenshot from the active page in the provided browser context.

    Parameters:
        browser_context: A Playwright browser context.

    Returns:
        A base64-encoded JPEG image string, or None if capturing fails.
    """
    playwright_browser = browser_context.browser.playwright_browser  # Ensure this is correct.
    if playwright_browser and playwright_browser.contexts:
        playwright_context = playwright_browser.contexts[0]
    else:
        return None
    pages = playwright_context.pages if playwright_context else None
    if pages:
        active_page = pages[0]
        for page in pages:
            if page.url != "about:blank":
                active_page = page
    else:
        return None
    try:
        screenshot = await active_page.screenshot(type='jpeg', quality=75, scale="css")
        encoded = base64.b64encode(screenshot).decode('utf-8')
        return encoded
    except Exception as e:
        return None

# ==============================================================
# END OF UTILS CODE
# ==============================================================

if __name__ == '__main__':
    # For quick testing purposes.
    print("Available Ollama Models:", model_names.get("ollama"))
