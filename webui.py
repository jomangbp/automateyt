#!/usr/bin/env python3
"""
webui.py - Browser Use WebUI with YouTube Functions

This file provides a complete and functional Gradio UI for controlling browser agents,
performing deep research, generating Markdown scripts, and includes YouTube-related functions.
It supports multiple LLM providers with dynamic model dropdown updates.
The dedicated YouTube tab performs deep research on a given topic and then uses the script
generator (from script_generator.py, using models defined in youtube_models.py) to create
a YouTube Shorts script. It also includes video upload functionality via the YouTube Data API.
"""

# ==============================================================
# IMPORTS
# ==============================================================

import base64
import os
import glob
import time
import asyncio
import argparse
import datetime
import pickle
import logging
from pathlib import Path
from typing import Dict, Optional
from uuid import uuid4

import gradio as gr
from playwright.async_api import async_playwright  # For browser automation

# YouTube API imports
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
import google_auth_oauthlib.flow
import google.auth.transport.requests

# Import modules for browser control and agent operations
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig, BrowserContextWindowSize

# Import custom modules from src
from src.utils.agent_state import AgentState
from src.utils import utils  # Ensure this contains the updated model_names with "deepseek-r1:1.5b" in ollama
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import (
    default_config,
    load_config_from_file,
    save_config_to_file,
    save_current_config,
    update_ui_from_config,
)
from src.utils.utils import get_latest_files, capture_screenshot, update_model_dropdown

# Import YouTube script generator and models
from src.utils.script_generator import generate_short_script as yt_generate_short_script
from src.utils.youtube_models import YouTubeShortScript, ScriptTone

# ==============================================================
# MONKEY-PATCH FOR CLEANING JSON OUTPUT
# ==============================================================

def clean_json_output(text: str) -> str:
    """
    Cleans a JSON string that may be wrapped in markdown code block markers.
    Removes leading/trailing backticks and the 'json' language specifier.
    """
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# Patch the parse_raw method of YouTubeShortScript to clean JSON output before parsing.
_original_parse_raw = YouTubeShortScript.parse_raw
def _patched_parse_raw(cls, b, *args, **kwargs):
    cleaned = clean_json_output(b)
    return _original_parse_raw(cleaned, *args, **kwargs)
YouTubeShortScript.parse_raw = classmethod(_patched_parse_raw)

# ==============================================================
# THEME MAP DEFINITION
# ==============================================================

theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base()
}

# ==============================================================
# GLOBAL VARIABLES
# ==============================================================

logger = logging.getLogger(__name__)

# Global browser and context objects for persistence between runs
_global_browser = None
_global_browser_context = None

# Global agent state instance to manage stop requests
_global_agent_state = AgentState()

# ==============================================================
# UTILITY FUNCTIONS (Outside Classes)
# ==============================================================

def generate_script(script_title, script_content):
    """
    Generates a Markdown file with the provided script content.
    Saves the generated script in the 'generated_scripts' directory.
    Returns the file path and the Markdown text for preview.
    """
    scripts_dir = "generated_scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{script_title.strip().replace(' ', '_')}_{timestamp}.md"
    filepath = os.path.join(scripts_dir, filename)
    markdown_text = f"# {script_title}\n\n{script_content}\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(markdown_text)
    return filepath, markdown_text

# ==============================================================
# AGENT CONTROL FUNCTIONS
# ==============================================================

async def stop_agent():
    """
    Requests the agent to stop and updates the UI with feedback.
    """
    global _global_agent_state, _global_browser_context, _global_browser
    try:
        _global_agent_state.request_stop()
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")
        return (
            message,
            gr.update(value="Stopping...", interactive=False),
            gr.update(interactive=False)
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            error_msg,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def stop_research_agent():
    """
    Requests the research agent to stop and updates the UI with feedback.
    """
    global _global_agent_state, _global_browser_context, _global_browser
    try:
        _global_agent_state.request_stop()
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")
        return (
            gr.update(value="Stopping...", interactive=False),
            gr.update(interactive=False)
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_browser_agent(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    """
    Runs the browser agent with the specified configuration.
    """
    global _global_agent_state
    _global_agent_state.clear_stop()

    try:
        if not enable_recording:
            save_recording_path = None

        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
        else:
            existing_videos = set()

        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key
        )

        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        latest_video = None
        if save_recording_path:
            new_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )
            if new_videos - existing_videos:
                latest_video = list(new_videos - existing_videos)[0]

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            latest_video,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',
            errors,
            '',
            '',
            None,
            None,
            None,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )

async def run_org_agent(
    llm,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    task,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    """
    Runs the original agent.
    """
    try:
        global _global_browser, _global_browser_context, _global_agent_state
        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        agent = Agent(
            task=task,
            llm=llm,
            use_vision=use_vision,
            browser=_global_browser,
            browser_context=_global_browser_context,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
        history = await agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
        agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_custom_agent(
    llm,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    """
    Runs the custom agent.
    """
    try:
        global _global_browser, _global_browser_context, _global_agent_state

        _global_agent_state.clear_stop()

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        if use_own_browser:
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        if _global_browser is None:
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        agent = CustomAgent(
            task=task,
            add_infos=add_infos,
            use_vision=use_vision,
            llm=llm,
            browser=_global_browser,
            browser_context=_global_browser_context,
            controller=controller,
            system_prompt_class=CustomSystemPrompt,
            agent_prompt_class=CustomAgentMessagePrompt,
            max_actions_per_step=max_actions_per_step,
            agent_state=_global_agent_state,
            tool_calling_method=tool_calling_method
        )
        history = await agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{agent.agent_id}.json")
        agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None

async def run_with_stream(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method
):
    """
    Runs the agent with streaming output.
    """
    global _global_agent_state
    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method
        )
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        yield [html_content] + list(result)
    else:
        try:
            _global_agent_state.clear_stop()
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method
                )
            )

            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            latest_videos = trace = history_file = None

            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent_state and _global_agent_state.is_stop_requested():
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),
                        gr.update(interactive=False)
                    ]
                    break
                else:
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        latest_videos,
                        trace,
                        history_file,
                        gr.update(value="Stop", interactive=True),
                        gr.update(interactive=True)
                    ]
                await asyncio.sleep(0.05)

            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, latest_videos, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                latest_videos = trace = history_file = None
            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                html_content,
                final_result,
                errors,
                model_actions,
                model_thoughts,
                latest_videos,
                trace,
                history_file,
                stop_button,
                run_button
            ]

        except Exception as e:
            import traceback
            yield [
                f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),
                gr.update(interactive=True)
            ]

async def close_global_browser():
    """
    Closes the global browser and context.
    """
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None

async def run_deep_search(
    research_task,
    max_search_iteration_input,
    max_query_per_iter_input,
    llm_provider,
    llm_model_name,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_vision,
    use_own_browser,
    headless
):
    """
    Runs a deep research task and returns the Markdown content and file path.
    """
    from src.utils.deep_research import deep_research
    global _global_agent_state

    _global_agent_state.clear_stop()

    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )
    markdown_content, file_path = await deep_research(
        research_task,
        llm,
        _global_agent_state,
        max_search_iterations=max_search_iteration_input,
        max_query_num=max_query_per_iter_input,
        use_vision=use_vision,
        headless=headless,
        use_own_browser=use_own_browser
    )

    return markdown_content, file_path, gr.update(value="Stop", interactive=True), gr.update(interactive=True)

# ==============================================================
# YOUTUBE SCRIPT GENERATION AND VIDEO UPLOAD FUNCTIONS
# ==============================================================

async def generate_youtube_script_with_research(
    topic: str,
    tone: str,
    target_duration: float = 60,
    llm_provider: str = "openai",
    llm_model_name: str = "gpt-4",
    llm_temperature: float = 0.7,
    llm_base_url: str = "",
    llm_api_key: str = "",
    **kwargs
) -> tuple[str, str]:
    """
    Generate a YouTube Short script using deep research as context.
    1) Perform deep research on the given topic.
    2) Augment the topic with the research report.
    3) Use the script generator to create a structured YouTube Short script.
    Returns the file path of the generated script and a Markdown preview.

    Note: The tone input must be one of the following:
          "informative", "entertaining", "educational", "dramatic", "humorous".
    """
    # 1) Run deep research
    research_markdown, research_file, _, _ = await run_deep_search(
        research_task=topic,
        max_search_iteration_input=kwargs.get("max_search_iterations", 3),
        max_query_per_iter_input=kwargs.get("max_query_per_iteration", 1),
        llm_provider=llm_provider,
        llm_model_name=llm_model_name,
        llm_temperature=llm_temperature,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        use_vision=kwargs.get("use_vision", False),
        use_own_browser=kwargs.get("use_own_browser", False),
        headless=kwargs.get("headless", True)
    )

    # 2) Augment the topic with the research context
    augmented_topic = f"{topic}\n\nResearch Report:\n{research_markdown}"

    # 3) Generate the YouTube Short script using the script generator
    script, script_path = await yt_generate_short_script(
        topic=augmented_topic,
        tone=ScriptTone(tone),
        target_duration=target_duration,
        llm_provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        api_key=llm_api_key,
        base_url=llm_base_url
    )

    # Build a Markdown preview of the generated script
    markdown_preview = f"""# {script.title}

**Tone:** {script.tone.value}  
**Target Duration:** {script.target_duration}s  

## Hook
{script.hook}

## Segments
{"".join([f"- (Duration: {seg.duration}s) {seg.text}\n" for seg in script.segments])}

## Hashtags
{' '.join(script.hashtags)}

## Call to Action
{script.call_to_action or 'None'}
"""
    return script_path, markdown_preview

def get_authenticated_service():
    """
    Authenticates and returns the YouTube Data API service object.
    Assumes 'client_secrets.json' is available and credentials are stored in 'token.pickle'.
    """
    scopes = ["https://www.googleapis.com/auth/youtube.upload"]
    client_secrets_file = "client_secrets.json"
    credentials = None

    if os.path.exists("token.pickle"):
        with open("token.pickle", "rb") as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(google.auth.transport.requests.Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                client_secrets_file, scopes
            )
            credentials = flow.run_console()
        with open("token.pickle", "wb") as token:
            pickle.dump(credentials, token)

    return build("youtube", "v3", credentials=credentials)

def upload_video_to_youtube(video_file, title, description, tags=None, categoryId="22", privacyStatus="private"):
    """
    Uploads a video to YouTube using the YouTube Data API.
    """
    try:
        youtube = get_authenticated_service()
        body = dict(
            snippet=dict(
                title=title,
                description=description,
                tags=tags.split(",") if tags else [],
                categoryId=categoryId
            ),
            status=dict(
                privacyStatus=privacyStatus
            )
        )
        media = MediaFileUpload(video_file, chunksize=-1, resumable=True)
        request = youtube.videos().insert(
            part="snippet,status",
            body=body,
            media_body=media
        )

        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                logger.info(f"Uploaded {int(status.progress() * 100)}%")
        logger.info("Upload Complete!")
        return response
    except HttpError as e:
        logger.error(f"An HTTP error {e.resp.status} occurred: {e.content}")
        return None
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return None

# ==============================================================
# UI CREATION
# ==============================================================

def create_ui(config, theme_name="Ocean"):
    """
    Creates the Gradio UI with multiple tabs including:
      - Agent Settings
      - LLM Configuration (with dynamic model update)
      - Browser Settings
      - Run Agent
      - Deep Research
      - Results
      - Recordings
      - Configuration
      - Dedicated YouTube Tab (performs deep research then generates a YouTube Short script)
    """
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    with gr.Blocks(title="Browser Use WebUI", theme=theme_map[theme_name], css=css) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"],
            )

        with gr.Tabs() as tabs:
            # 1) Agent Settings
            with gr.TabItem("⚙️ Agent Settings", id="1"):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value=config['agent_type'],
                        info="Select the type of agent to use",
                    )
                    with gr.Column():
                        max_steps = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=config['max_steps'],
                            step=1,
                            label="Max Run Steps",
                            info="Maximum number of steps the agent will take",
                        )
                        max_actions_per_step = gr.Slider(
                            minimum=1,
                            maximum=20,
                            value=config['max_actions_per_step'],
                            step=1,
                            label="Max Actions per Step",
                            info="Maximum number of actions the agent will take per step",
                        )
                    with gr.Column():
                        use_vision = gr.Checkbox(
                            label="Use Vision",
                            value=config['use_vision'],
                            info="Enable visual processing capabilities",
                        )
                        tool_calling_method = gr.Dropdown(
                            label="Tool Calling Method",
                            value=config['tool_calling_method'],
                            interactive=True,
                            allow_custom_value=True,
                            choices=["auto", "json_schema", "function_calling"],
                            info="Tool Calls Function Name",
                            visible=False
                        )

            # 2) LLM Configuration
            with gr.TabItem("🔧 LLM Configuration", id="2"):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        choices=list(utils.model_names.keys()),
                        label="LLM Provider",
                        value=config['llm_provider'],
                        info="Select your preferred language model provider"
                    )
                    llm_model_name = gr.Dropdown(
                        label="Model Name",
                        choices=utils.model_names.get(config['llm_provider'], []),
                        value=config['llm_model_name'],
                        interactive=True,
                        allow_custom_value=True,
                        info="Select a model from the dropdown or type a custom model name"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=config['llm_temperature'],
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            value=config['llm_base_url'],
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=config['llm_api_key'],
                            info="Your API key (leave blank to use .env)"
                        )

                llm_provider.change(
                    fn=update_model_dropdown,
                    inputs=[llm_provider, llm_api_key, llm_base_url],
                    outputs=llm_model_name
                )

            # 3) Browser Settings
            with gr.TabItem("🌐 Browser Settings", id="3"):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=config['use_own_browser'],
                            info="Use your existing browser instance",
                        )
                        keep_browser_open = gr.Checkbox(
                            label="Keep Browser Open",
                            value=config['keep_browser_open'],
                            info="Keep Browser Open between Tasks",
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=config['headless'],
                            info="Run browser without GUI",
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=config['disable_security'],
                            info="Disable browser security features",
                        )
                        enable_recording = gr.Checkbox(
                            label="Enable Recording",
                            value=config['enable_recording'],
                            info="Enable saving browser recordings",
                        )

                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=config['window_w'],
                            info="Browser window width",
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=config['window_h'],
                            info="Browser window height",
                        )

                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value=config['save_recording_path'],
                        info="Path to save browser recordings",
                        interactive=True
                    )

                    save_trace_path = gr.Textbox(
                        label="Trace Path",
                        placeholder="e.g. ./tmp/traces",
                        value=config['save_trace_path'],
                        info="Path to save Agent traces",
                        interactive=True
                    )

                    save_agent_history_path = gr.Textbox(
                        label="Agent History Save Path",
                        placeholder="e.g., ./tmp/agent_history",
                        value=config['save_agent_history_path'],
                        info="Directory where agent history should be saved.",
                        interactive=True
                    )

            # 4) Run Agent
            with gr.TabItem("🤖 Run Agent", id="4"):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value=config['task'],
                    info="Describe what you want the agent to do",
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

                with gr.Row():
                    browser_view = gr.HTML(
                        value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
                        label="Live Browser View",
                    )

            # 5) Deep Research
            with gr.TabItem("🧐 Deep Research", id="5"):
                research_task_input = gr.Textbox(
                    label="Research Task",
                    lines=5,
                    value="Compose a report on the use of Reinforcement Learning for training Large Language Models...",
                )
                with gr.Row():
                    max_search_iteration_input = gr.Number(label="Max Search Iteration", value=3, precision=0)
                    max_query_per_iter_input = gr.Number(label="Max Query per Iteration", value=1, precision=0)
                with gr.Row():
                    research_button = gr.Button("▶️ Run Deep Research", variant="primary", scale=2)
                    stop_research_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

                markdown_output_display = gr.Markdown(label="Research Report")
                markdown_download = gr.File(label="Download Research Report")

                stop_research_button.click(
                    fn=stop_research_agent,
                    inputs=[],
                    outputs=[stop_research_button, research_button],
                )
                research_button.click(
                    fn=run_deep_search,
                    inputs=[
                        research_task_input, max_search_iteration_input, max_query_per_iter_input,
                        llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_vision, use_own_browser, headless
                    ],
                    outputs=[markdown_output_display, markdown_download, stop_research_button, research_button]
                )

            # 6) Results
            with gr.TabItem("📊 Results", id="6"):
                with gr.Group():
                    recording_display = gr.Video(label="Latest Recording")
                    gr.Markdown("### Results")

                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(label="Final Result", lines=3, show_label=True)
                        with gr.Column():
                            errors_output = gr.Textbox(label="Errors", lines=3, show_label=True)

                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(label="Model Actions", lines=3, show_label=True)
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(label="Model Thoughts", lines=3, show_label=True)

                    trace_file = gr.File(label="Trace File")
                    agent_history_file = gr.File(label="Agent History")

                stop_button.click(
                    fn=stop_agent,
                    inputs=[],
                    outputs=[errors_output, stop_button, run_button],
                )

                run_button.click(
                    fn=run_with_stream,
                    inputs=[
                        agent_type, llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                        save_recording_path, save_agent_history_path, save_trace_path,
                        enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step, tool_calling_method
                    ],
                    outputs=[
                        browser_view,
                        final_result_output,
                        errors_output,
                        model_actions_output,
                        model_thoughts_output,
                        recording_display,
                        trace_file,
                        agent_history_file,
                        stop_button,
                        run_button
                    ],
                )

            # 7) Recordings
            with gr.TabItem("🎥 Recordings", id="7"):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + \
                                 glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
                    recordings.sort(key=os.path.getctime)
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))
                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    value=list_recordings(config['save_recording_path']),
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")
                refresh_button.click(
                    fn=list_recordings,
                    inputs=save_recording_path,
                    outputs=recordings_gallery
                )

            # 8) Configuration
            with gr.TabItem("📁 Configuration", id="8"):
                with gr.Group():
                    config_file_input = gr.File(
                        label="Load Config File",
                        file_types=[".pkl"],
                        interactive=True
                    )
                    load_config_button = gr.Button("Load Existing Config From File", variant="primary")
                    save_config_button = gr.Button("Save Current Config", variant="primary")
                    config_status = gr.Textbox(label="Status", lines=2, interactive=False)

                load_config_button.click(
                    fn=update_ui_from_config,
                    inputs=[config_file_input],
                    outputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security, enable_recording,
                        window_w, window_h, save_recording_path, save_trace_path, save_agent_history_path,
                        task, config_status
                    ]
                )

                save_config_button.click(
                    fn=save_current_config,
                    inputs=[
                        agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                        llm_provider, llm_model_name, llm_temperature, llm_base_url, llm_api_key,
                        use_own_browser, keep_browser_open, headless, disable_security,
                        enable_recording, window_w, window_h, save_recording_path, save_trace_path,
                        save_agent_history_path, task,
                    ],
                    outputs=[config_status]
                )

            # 9) Dedicated YouTube Tab
            with gr.TabItem("📹 YouTube", id="10"):
                gr.Markdown("## YouTube Functions")

                with gr.Accordion("Generate YouTube Short Script with Research", open=True):
                    yt_topic = gr.Textbox(label="Script Topic", placeholder="Enter a topic...", value="My YouTube Short")
                    yt_tone = gr.Dropdown(
                        label="Script Tone",
                        choices=[tone.value for tone in ScriptTone],
                        value=ScriptTone.ENTERTAINING.value
                    )
                    yt_duration = gr.Slider(label="Target Duration (seconds)", minimum=15, maximum=60, step=1, value=60)
                    yt_generate_btn = gr.Button("Generate Script", variant="primary")
                    yt_script_file = gr.File(label="Download Generated YouTube Script")
                    yt_script_preview = gr.Markdown(label="Script Preview")

                    yt_generate_btn.click(
                        fn=generate_youtube_script_with_research,
                        inputs=[
                            yt_topic,
                            yt_tone,
                            yt_duration,
                            llm_provider,
                            llm_model_name,
                            llm_temperature,
                            llm_base_url,
                            llm_api_key
                        ],
                        outputs=[yt_script_file, yt_script_preview]
                    )

                with gr.Accordion("Upload Video to YouTube", open=True):
                    video_file = gr.File(label="Video File")
                    yt_video_title = gr.Textbox(label="Video Title", placeholder="Enter video title...", value="My Video")
                    yt_video_description = gr.Textbox(label="Video Description", lines=3, placeholder="Enter video description...")
                    yt_video_tags = gr.Textbox(label="Video Tags", placeholder="Enter comma separated tags", value="tag1, tag2")
                    yt_upload_btn = gr.Button("Upload Video", variant="primary")
                    yt_upload_result = gr.JSON(label="Upload Result")

                    yt_upload_btn.click(
                        fn=upload_video_to_youtube,
                        inputs=[video_file, yt_video_title, yt_video_description, yt_video_tags],
                        outputs=[yt_upload_result]
                    )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

    return demo

# ==============================================================
# MAIN FUNCTION
# ==============================================================

def main():
    """
    Main function to parse command-line arguments and launch the Gradio UI.
    """
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()
    demo = create_ui(config_dict, theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == '__main__':
    main()
