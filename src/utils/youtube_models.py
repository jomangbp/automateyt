from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class ScriptTone(str, Enum):
    INFORMATIVE = "informative"
    ENTERTAINING = "entertaining"
    EDUCATIONAL = "educational"
    DRAMATIC = "dramatic"
    HUMOROUS = "humorous"

class ScriptSegment(BaseModel):
    text: str = Field(..., description="The actual content/dialogue for this segment")
    duration: float = Field(..., description="Duration in seconds for this segment")
    background_music: Optional[str] = Field(None, description="Background music suggestion")
    visual_effects: Optional[List[str]] = Field(default_factory=list, description="List of visual effects")

class YouTubeShortScript(BaseModel):
    title: str = Field(..., description="Title of the YouTube Short")
    tone: ScriptTone = Field(..., description="Overall tone of the script")
    target_duration: float = Field(..., ge=1, le=60, description="Target duration in seconds")
    hook: str = Field(..., description="Opening hook to capture viewer attention")
    segments: List[ScriptSegment] = Field(..., description="Script segments")
    hashtags: List[str] = Field(default_factory=list, description="Relevant hashtags")
    call_to_action: Optional[str] = Field(None, description="Call to action at the end")