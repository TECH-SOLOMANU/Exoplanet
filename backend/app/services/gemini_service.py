"""
Gemini 2.5 Integration Service for NASA Exoplanet Detection Platform
NASA Space Apps Challenge 2025
"""

import os
import json
import asyncio
from typing import Dict, List, Optional
import google.generativeai as genai
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GeminiExoplanetAssistant:
    """
    Advanced AI assistant powered by Gemini 2.5 for exoplanet education and research
    """
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_AI_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        else:
            self.model = None
            logger.warning("Gemini API key not found. Using fallback responses.")
        
        # Context for exoplanet domain
        self.system_context = """
        You are an expert exoplanet education assistant for a NASA Space Apps Challenge project.
        Your role is to help students, researchers, and enthusiasts learn about exoplanets.
        
        Key areas you excel at:
        - Exoplanet detection methods (transit, radial velocity, direct imaging, microlensing)
        - NASA missions (Kepler, K2, TESS, James Webb Space Telescope)
        - Astronomical concepts (habitable zones, stellar types, planetary formation)
        - Data analysis and machine learning in astronomy
        - Recent discoveries and their significance
        
        Guidelines:
        1. Adapt explanations to user's apparent knowledge level
        2. Use analogies and examples for complex concepts
        3. Encourage further exploration and learning
        4. Reference real NASA data and missions when relevant
        5. Be enthusiastic about astronomical discoveries
        6. Provide practical examples and interactive suggestions
        
        Always aim to inspire curiosity about space and science!
        """
    
    async def generate_response(self, message: str, user_level: str = "beginner", context: Dict = None) -> Dict:
        """
        Generate intelligent response using Gemini 2.5
        """
        if not self.model:
            return self._get_fallback_response(message, user_level)
        
        try:
            # Build enhanced prompt with context
            prompt = self._build_enhanced_prompt(message, user_level, context)
            
            # Generate response with Gemini
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=1000,
                )
            )
            
            # Parse and enhance response
            return self._parse_gemini_response(response.text, message)
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return self._get_fallback_response(message, user_level)
    
    def _build_enhanced_prompt(self, message: str, user_level: str, context: Dict = None) -> str:
        """
        Build context-aware prompt for Gemini
        """
        prompt_parts = [
            self.system_context,
            f"\nUser Level: {user_level}",
            f"\nUser Question: {message}"
        ]
        
        # Add contextual information if available
        if context:
            if context.get('current_page'):
                prompt_parts.append(f"\nUser is currently viewing: {context['current_page']}")
            
            if context.get('recent_data'):
                prompt_parts.append(f"\nRecent platform data: {context['recent_data']}")
            
            if context.get('conversation_history'):
                prompt_parts.append(f"\nRecent conversation: {context['conversation_history'][-3:]}")
        
        prompt_parts.extend([
            "\nPlease provide a helpful, engaging response that:",
            "1. Directly answers the user's question",
            "2. Includes relevant examples or analogies",
            "3. Suggests related topics to explore",
            "4. Encourages hands-on learning when appropriate",
            "\nResponse:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_gemini_response(self, response_text: str, original_question: str) -> Dict:
        """
        Parse and enhance Gemini response with additional features
        """
        # Generate follow-up suggestions based on the response
        suggestions = self._generate_smart_suggestions(response_text, original_question)
        
        return {
            "message": response_text,
            "type": "gemini",
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
            "confidence": "high",
            "source": "Gemini 2.5"
        }
    
    def _generate_smart_suggestions(self, response: str, question: str) -> List[str]:
        """
        Generate intelligent follow-up suggestions
        """
        # Analyze the response to suggest relevant follow-ups
        suggestions = []
        
        # Topic-based suggestions
        if any(word in response.lower() for word in ['kepler', 'transit']):
            suggestions.append("Tell me more about the Kepler mission")
            suggestions.append("How does the transit method work?")
        
        if any(word in response.lower() for word in ['habitable', 'goldilocks']):
            suggestions.append("What makes a planet habitable?")
            suggestions.append("Show me Earth-like exoplanets")
        
        if any(word in response.lower() for word in ['tess', 'jwst', 'telescope']):
            suggestions.append("What is the James Webb Space Telescope finding?")
            suggestions.append("How do space telescopes detect exoplanets?")
        
        if any(word in response.lower() for word in ['machine learning', 'ai', 'algorithm']):
            suggestions.append("How does AI help find exoplanets?")
            suggestions.append("Show me the ML model predictions")
        
        # Default suggestions if none match
        if not suggestions:
            suggestions = [
                "What are the most exciting recent discoveries?",
                "How can I contribute to exoplanet research?",
                "Explain this in simpler terms",
                "Show me real NASA data"
            ]
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    def _get_fallback_response(self, message: str, user_level: str) -> Dict:
        """
        Fallback responses when Gemini is not available
        """
        message_lower = message.lower()
        
        # Enhanced fallback responses
        if any(word in message_lower for word in ['what', 'define', 'explain']):
            if 'exoplanet' in message_lower:
                response = """
                🌟 Exoplanets are planets that orbit stars outside our solar system! 
                
                Think of it this way: just like Earth orbits our Sun, exoplanets orbit other stars in the universe. We've discovered over 5,000 confirmed exoplanets so far!
                
                The exciting part? Some might be habitable worlds where life could exist. NASA missions like Kepler and TESS have revolutionized how we find these distant worlds.
                
                *This response is using offline mode. Connect Gemini API for more advanced explanations!*
                """
                suggestions = [
                    "How do we detect exoplanets?",
                    "Show me habitable exoplanets",
                    "Tell me about NASA missions",
                    "What makes a planet habitable?"
                ]
            
            elif 'transit' in message_lower:
                response = """
                🔭 The transit method is like watching a shadow! 
                
                When an exoplanet passes in front of its star (from our perspective), it blocks a tiny bit of the star's light. This creates a small, temporary dip in brightness that we can measure.
                
                It's similar to how a bird flying in front of a streetlight creates a shadow. The Kepler Space Telescope used this method to discover thousands of exoplanets!
                
                *This response is using offline mode. Connect Gemini API for detailed explanations!*
                """
                suggestions = [
                    "What other detection methods exist?",
                    "Tell me about Kepler discoveries",
                    "How accurate is the transit method?",
                    "Show me a light curve example"
                ]
            
            else:
                response = """
                🚀 Great question! I'd love to give you a detailed explanation.
                
                For the most comprehensive and personalized answers about exoplanets, astronomy, and NASA missions, we recommend connecting the Gemini AI assistant.
                
                In the meantime, feel free to explore our interactive dashboard with real NASA data!
                
                *This response is using offline mode. Connect Gemini API for advanced AI assistance!*
                """
                suggestions = [
                    "What are exoplanets?",
                    "How do we find them?",
                    "Show me recent discoveries",
                    "Explore the dashboard"
                ]
        
        elif any(word in message_lower for word in ['how', 'method', 'detect']):
            response = """
            🔬 There are several amazing ways we detect exoplanets:
            
            1. **Transit Method**: Watch for tiny dips in starlight
            2. **Radial Velocity**: Measure star's wobble from gravitational pull
            3. **Direct Imaging**: Actually photograph the planet (rare!)
            4. **Microlensing**: Use gravity as a cosmic magnifying glass
            
            Each method reveals different types of planets and helps us build a complete picture of planetary systems!
            
            *This response is using offline mode. Connect Gemini API for detailed technical explanations!*
            """
            suggestions = [
                "Explain the transit method",
                "What is radial velocity?",
                "Show me detection examples",
                "Which method is most successful?"
            ]
        
        else:
            response = """
            🌌 That's an interesting question about exoplanets and astronomy!
            
            I'm currently running in offline mode with basic responses. For the most comprehensive, personalized, and up-to-date information about exoplanets, consider connecting the Gemini AI assistant.
            
            You can also explore our interactive features:
            - Browse real NASA exoplanet data
            - Try our AI prediction models
            - Visualize light curves and discoveries
            
            *Connect Gemini API for advanced AI-powered conversations!*
            """
            suggestions = [
                "What are exoplanets?",
                "How are they discovered?",
                "Show me the latest findings",
                "Explore NASA data"
            ]
        
        return {
            "message": response,
            "type": "fallback",
            "suggestions": suggestions,
            "timestamp": datetime.now().isoformat(),
            "confidence": "medium",
            "source": "Offline Assistant"
        }
    
    async def get_contextual_suggestions(self, current_page: str = None) -> List[str]:
        """
        Get smart suggestions based on current context
        """
        if current_page == "dashboard":
            return [
                "Explain these statistics to me",
                "What do the charts show?",
                "How is this data collected?",
                "Tell me about recent discoveries"
            ]
        elif current_page == "predictions":
            return [
                "How does the AI model work?",
                "What features are most important?",
                "How accurate are these predictions?",
                "Can I upload my own data?"
            ]
        elif current_page == "about":
            return [
                "Tell me more about this project",
                "How can I contribute to research?",
                "What makes this platform special?",
                "Connect me with NASA resources"
            ]
        else:
            return [
                "What are exoplanets?",
                "How do we detect them?",
                "Show me recent discoveries",
                "Guide me through the platform"
            ]

# Singleton instance
gemini_assistant = GeminiExoplanetAssistant()