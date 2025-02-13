import os
from typing import Optional
from uuid import uuid4
from langchain_core.messages import SystemMessage, HumanMessage
from .youtube_models import YouTubeShortScript, ScriptTone
from .utils import get_llm_model

async def generate_short_script(
    topic: str,
    tone: ScriptTone,
    target_duration: float = 60,
    llm_provider: str = "gemini",
    **kwargs
) -> tuple[YouTubeShortScript, str]:
    """
    Generate a YouTube Short script using AI assistance
    """
    system_prompt = """You are a professional YouTube Shorts script writer. Your task is to create engaging, 
    concise scripts that capture attention within the first few seconds and maintain viewer interest throughout.
    
    The output must be a valid JSON object matching the YouTubeShortScript schema, containing:
    - title: Catchy title for the Short
    - tone: Overall tone (from provided ScriptTone)
    - target_duration: Total duration in seconds
    - hook: Attention-grabbing opening line
    - segments: List of segments with text, duration, and optional effects
    - hashtags: Relevant hashtags
    - call_to_action: Optional call to action
    
    Important guidelines:
    1. Keep the total duration within the specified target
    2. Make the hook compelling and direct
    3. Structure segments for smooth flow
    4. Include relevant hashtags
    5. Maintain consistent tone throughout
    """

    llm = get_llm_model(
        provider=llm_provider,
        model_name=kwargs.get("model_name", "gemini-pro"),
        temperature=kwargs.get("temperature", 0.7),
        api_key=kwargs.get("api_key", "")
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Create a {tone.value} YouTube Short script about: {topic}. "
                            f"Target duration: {target_duration} seconds")
    ]

    response = llm.invoke(messages)
    
    # Parse the response into our Pydantic model
    script = YouTubeShortScript.parse_raw(response.content)
    
    # Save the script as markdown
    script_path = os.path.join("./tmp/scripts", f"{uuid4()}.md")
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    
    markdown_content = f"""# {script.title}

## Hook
{script.hook}

## Segments
{chr(10).join([f'### Segment {i+1} ({seg.duration}s)\n{seg.text}\n' 
               for i, seg in enumerate(script.segments)])}

## Hashtags
{' '.join(script.hashtags)}

## Call to Action
{script.call_to_action or 'None'}
"""

    with open(script_path, "w") as f:
        f.write(markdown_content)

    return script, script_path