"""
Safety analyzer module for detecting safety issues in garbage collection video frames.
"""
import base64
import io
import json
import cv2
from openai import OpenAI

from src.utils.config import get_openai_api_key


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
        Identify any safety issues related to collecting trash from trash cans visible in the frame.
        
        Examples of safety issues to look for:
        - Fire or smoke coming from trash cans
        - Fire or smoke coming from the garbage truck itself or its collector crane.
        - Hazardous materials visible (chemical containers, batteries, etc.)
        - Blocked access to trash cans
        - Unstable or damaged trash cans
        - Dangerous objects protruding from trash cans
        - People or animals too close to the collection area
        - Weather-related hazards (ice, flooding, etc.)
        
        If any safety issues are detected, return a JSON array with the following structure for each issue:
        [
            {
                "issue_type": "fire", 
                "confidence": 0.95,
                "location": "left trash can",
                "description": "Visible flames coming from the trash can on the left side of the frame"
            }
        ]
        
        If no safety issues are detected, return an empty JSON array: []
        """
        
        # Make the API call
        response = self.client.chat.completions.create(
            model="gpt-4.1",
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
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        
        # Extract and parse the response
        result_text = response.choices[0].message.content
        try:
            # The API returns JSON inside a JSON response, so we need to parse it
            result = json.loads(result_text)
            
            # If the result doesn't have the 'safety_issues' key, it's likely a direct array
            if isinstance(result, dict) and not result.get('safety_issues'):
                # Try to find any array in the response
                for key, value in result.items():
                    if isinstance(value, list):
                        return {"safety_issues": value}
                
                # If no array found, return empty list
                return {"safety_issues": []}
            
            # Return the parsed result
            return result
        except json.JSONDecodeError:
            # Handle case where response is not valid JSON
            return {"safety_issues": [], "error": "Failed to parse API response"}
        except Exception as e:
            return {"safety_issues": [], "error": str(e)} 