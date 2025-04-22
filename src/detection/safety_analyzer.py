"""
Safety analyzer module for detecting safety issues in garbage collection video frames.
"""
import base64
import io
import cv2
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

from src.utils.config import get_openai_api_key


class SafetyIssue(BaseModel):
    """Model representing a safety issue detected in a frame."""
    issue_type: str
    location: str
    description: str


class SafetyAnalysisResult(BaseModel):
    """Model representing the result of a safety analysis."""
    safety_issues: List[SafetyIssue] = Field(default_factory=list)
    error: Optional[str] = None


class SafetyAnalyzer:
    """
    A class to analyze video frames for safety issues using OpenAI.
    """
    
    def __init__(self):
        """Initialize the safety analyzer with OpenAI client."""
        api_key = get_openai_api_key()
        self.client = OpenAI(api_key=api_key)
    
    def analyze_frame(self, frame):
        """
        Analyze a frame for safety issues related to garbage collection.
        
        Args:
            frame (numpy.ndarray): The video frame to analyze
        
        Returns:
            dict: Analysis results with detected safety issues
        """
        # Convert the frame to a base64 string for the API
        _, buffer = cv2.imencode('.jpg', frame)
        img_bytes = io.BytesIO(buffer)
        base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
        
        # Prepare the prompt for the API
        prompt = """
        Analyze this image from a garbage truck's point of view.
        Identify obvious and clear safety issues related to collecting trash from trash cans visible in the frame.
        
        Examples of safety issues to look for:
        - Fire or smoke coming from trash cans
        - Fire or smoke coming from the garbage truck itself or its collector crane.
        - Hazardous materials visible (chemical containers, batteries, etc.)
        - Dangerous and sharp objects protruding from trash cans
        - People or animals too close to the collection area
        - Weather-related hazards (ice, flooding, etc.)

        Issues like a vehicle being too close to the collection area are not safety issues (unless they are extremely close.)
        
        If no safety issues are detected, return an empty array.
        """
        
        try:
            # Make the API call with structured output using parse
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format=SafetyAnalysisResult,
                max_tokens=1000
            )
            
            # Return the parsed result directly
            return completion.choices[0].message.parsed.model_dump()
        except Exception as e:
            return SafetyAnalysisResult(safety_issues=[], error=str(e)).model_dump() 