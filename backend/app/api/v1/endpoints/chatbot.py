"""
NASA Exoplanet Chatbot API endpoints with Gemini AI integration
NASA Space Apps Challenge 2025 - Updated with Gemini REST API
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import logging
import asyncio
import random
import os
from app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Import Gemini AI (using REST API approach)
try:
    import google.generativeai as genai
    # We're using REST API directly, so package import isn't critical
    GEMINI_AVAILABLE = True  # Enable Gemini via REST API
    logger.info("Gemini enabled via REST API - bypassing package version limitations")
except ImportError as e:
    GEMINI_AVAILABLE = True  # Still enable since we use REST API
    logger.info(f"Google Generative AI package not installed: {e}. Using Gemini REST API directly.")
except Exception as e:
    GEMINI_AVAILABLE = True  # Still enable since we use REST API
    logger.info(f"Package configuration warning: {e}. Using Gemini REST API directly.")

# Initialize router
router = APIRouter(tags=["chatbot"])
logger = logging.getLogger(__name__)

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    level: str = "beginner"

class ChatResponse(BaseModel):
    response: str
    suggestions: List[str] = []

class UserLevel(BaseModel):
    level: str

# Educational content database
EXOPLANET_KNOWLEDGE = {
    "basics": {
        "what are exoplanets": """
        🌍 **Exoplanets** (or extrasolar planets) are planets that orbit stars outside our solar system!
        
        **Key Facts:**
        • Over 5,000 exoplanets have been discovered
        • They range from rocky Earth-like worlds to gas giants
        • Some might have conditions suitable for life
        • The first exoplanet was discovered in 1995
        
        *Fun fact: There are likely billions of exoplanets in our galaxy alone!*
        """,
        
        "detection methods": """
        🔭 **How We Find Exoplanets:**
        
        **1. Transit Method** (Most common)
        • Watch for tiny dips in starlight when a planet passes in front
        • Kepler and TESS missions use this method
        
        **2. Radial Velocity**
        • Detect the "wobble" of stars caused by orbiting planets
        
        **3. Direct Imaging**
        • Take actual pictures of exoplanets (very rare!)
        
        **4. Gravitational Microlensing**
        • Use gravity as a natural telescope
        """,
        
        "kepler mission": """
        🚀 **Kepler Space Telescope (2009-2018)**
        
        **Amazing Achievements:**
        • Discovered over 2,600 confirmed exoplanets
        • Found planets in the "habitable zone"
        • Proved that rocky planets are common
        • Showed us planetary systems very different from ours
        
        **How it worked:**
        • Stared at 150,000 stars continuously
        • Measured tiny brightness changes
        • Discovered planets as small as Earth!
        """,
        
        "habitable zone": """
        🌡️ **The Goldilocks Zone**
        
        The habitable zone is the "just right" distance from a star where:
        • Water can exist as a liquid
        • Not too hot (water boils away)
        • Not too cold (water freezes)
        
        **Examples:**
        • Earth is in our Sun's habitable zone
        • Kepler-452b: "Earth's cousin"
        • TRAPPIST-1 system has 3 planets in habitable zone!
        """
    },
    
    "intermediate": {
        "transit photometry": """
        📊 **Transit Photometry Deep Dive**
        
        **The Science:**
        • Planet blocks ~0.01-1% of starlight
        • Transit depth = (Planet radius / Star radius)²
        • Transit duration depends on orbital period and star size
        
        **What we learn:**
        • Planet size and density
        • Orbital period and distance
        • Sometimes atmospheric composition
        
        **Challenges:**
        • False positives from binary stars
        • Need multiple transits to confirm
        • Requires very precise measurements
        """,
        
        "atmospheric analysis": """
        🌫️ **Exoplanet Atmospheres**
        
        **Transmission Spectroscopy:**
        • Light filters through planet's atmosphere during transit
        • Different molecules absorb specific wavelengths
        • We can detect water vapor, methane, oxygen!
        
        **Key Discoveries:**
        • Hot Jupiters with water vapor
        • Super-Earths with hydrogen atmospheres
        • Evidence of clouds and hazes
        
        **Future:** JWST is revolutionizing atmospheric studies!
        """,
        
        "tidal locking": """
        🔒 **Tidally Locked Worlds**
        
        **What is it?**
        • One side always faces the star (like our Moon to Earth)
        • Common for close-in exoplanets
        
        **Consequences:**
        • Permanent day side (very hot)
        • Permanent night side (very cold)
        • Potential for extreme weather patterns
        
        **Habitability:**
        • Terminator zone might be habitable
        • Atmospheric circulation could redistribute heat
        """
    },
    
    "advanced": {
        "machine learning": """
        🤖 **AI in Exoplanet Discovery**
        
        **Our Platform Uses:**
        • **XGBoost**: For tabular data classification
        • **CNNs**: For light curve analysis
        • **Ensemble Methods**: Combining multiple models
        
        **Why ML?**
        • Process millions of light curves automatically
        • Find subtle patterns humans might miss
        • Reduce false positives
        • Enable real-time discovery
        
        **Recent Breakthroughs:**
        • Neural networks found planets in Kepler data
        • Deep learning improved transit detection
        • Automated vetting of planet candidates
        """,
        
        "statistical validation": """
        📈 **Planet Validation Statistics**
        
        **False Positive Sources:**
        • Eclipsing binary stars
        • Background stars
        • Stellar activity
        
        **Validation Methods:**
        • Statistical validation (this platform!)
        • Follow-up observations
        • Centroid analysis
        • Multi-color photometry
        
        **Success Rates:**
        • Kepler Objects of Interest: ~85% validated
        • TESS candidates: ongoing validation
        """
    }
}

# Quick suggestions based on level
SUGGESTIONS = {
    "beginner": [
        "What are exoplanets?",
        "How do we detect exoplanets?",
        "Tell me about the Kepler mission",
        "What is the habitable zone?"
    ],
    "intermediate": [
        "Explain transit photometry",
        "How do we study exoplanet atmospheres?",
        "What is tidal locking?",
        "Show me detection statistics"
    ],
    "advanced": [
        "How does machine learning help find exoplanets?",
        "Explain statistical validation methods",
        "What are the latest AI discoveries?",
        "Tell me about ensemble models"
    ]
}

def get_offline_response(message: str, level: str = "beginner") -> ChatResponse:
    """Generate educational response based on message content and user level"""
    
    message_lower = message.lower()
    
    # Map user levels to knowledge base sections
    level_mapping = {
        "beginner": "basics",
        "intermediate": "intermediate", 
        "advanced": "advanced"
    }
    
    # Search for relevant topics
    response_text = ""
    found_topic = False
    
    # Get mapped level for knowledge base lookup
    mapped_level = level_mapping.get(level, "basics")
    knowledge_base = EXOPLANET_KNOWLEDGE.get(mapped_level, EXOPLANET_KNOWLEDGE["basics"])
    
    for topic, content in knowledge_base.items():
        if any(keyword in message_lower for keyword in topic.split()):
            response_text = content
            found_topic = True
            break
    
    # If no specific topic found, search all levels
    if not found_topic:
        for level_content in EXOPLANET_KNOWLEDGE.values():
            for topic, content in level_content.items():
                if any(keyword in message_lower for keyword in topic.split()):
                    response_text = content
                    found_topic = True
                    break
            if found_topic:
                break
    
    # General responses for common questions
    if not found_topic:
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            response_text = """
            👋 Hello! I'm your NASA Exoplanet Assistant! 
            
            I'm here to help you learn about exoplanets and their discovery. I can explain:
            • Exoplanet basics and detection methods
            • NASA missions like Kepler and TESS
            • How our AI models work
            • Latest discoveries and research
            
            What would you like to explore first?
            """
        elif any(word in message_lower for word in ['help', 'what can you do']):
            response_text = f"""
            🤖 **I can help you with:**
            
            **🌍 Exoplanet Education:**
            • Basic concepts and definitions
            • Detection methods and missions
            • Fun facts and latest discoveries
            
            **🔬 Scientific Details:**
            • Transit photometry and analysis
            • Atmospheric studies
            • Statistical validation
            
            **🤖 AI & Machine Learning:**
            • How our models work
            • Ensemble methods
            • Real-time predictions
            
            **Current Level:** {level.title()}
            Ask me anything about exoplanets!
            """
        elif any(word in message_lower for word in ['nasa', 'space apps', 'challenge']):
            response_text = """
            🏆 **NASA Space Apps Challenge 2025**
            
            This platform was built for the NASA Space Apps Challenge with the goal of:
            • Making exoplanet science accessible to everyone
            • Using AI to accelerate discovery
            • Providing hands-on learning experiences
            • Connecting students with real NASA data
            
            Our team implemented advanced ML models trained on real Kepler, K2, and TESS data!
            """
        else:
            response_text = f"""
            🤔 I'd love to help you learn about exoplanets! 
            
            Try asking me about:
            • "What are exoplanets?"
            • "How do we detect them?"
            • "Tell me about Kepler mission"
            • "How does AI help in discovery?"
            
            Or choose from the suggestions below! 
            (Your current level: {level.title()})
            """
    
    # Get suggestions based on level
    suggestions = SUGGESTIONS.get(level, SUGGESTIONS["beginner"])
    
    return ChatResponse(
        response=response_text.strip(),
        suggestions=suggestions
    )

async def get_gemini_response(message: str, level: str = "beginner") -> ChatResponse:
    """
    Get response from Gemini AI using REST API (compatible with new API keys)
    """
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == "your_gemini_api_key_here":
        logger.warning("Gemini API key not configured, using fallback response")
        return get_offline_response(message, level)
    
    try:
        import aiohttp
        import json
        
        # Create a specialized prompt for exoplanet education
        system_prompt = f"""
        You are an expert NASA exoplanet educator and assistant for the NASA Space Apps Challenge 2025. 
        Your role is to teach about exoplanets in an engaging, accurate way.
        
        User Level: {level}
        - Beginner: Use simple terms, analogies, fun facts
        - Intermediate: Include scientific details, methods, missions  
        - Advanced: Discuss research papers, technical details, cutting-edge discoveries
        
        Guidelines:
        - Always be enthusiastic about space and exoplanets 🚀
        - Use emojis to make learning fun
        - Include specific examples of real exoplanets when relevant
        - Mention NASA missions (Kepler, TESS, K2) when appropriate
        - Keep responses concise but informative (under 300 words)
        - End with a follow-up question to encourage learning
        
        Topic: Exoplanets, detection methods, NASA missions, astrobiology, planetary science
        """
        
        full_prompt = f"{system_prompt}\n\nUser Question: {message}\n\nProvide an educational response:"
        
        # Use Gemini REST API directly
        api_key = settings.GEMINI_API_KEY
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={api_key}"
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": full_prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 400
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=15) as response:
                if response.status == 200:
                    result = await response.json()
                    if "candidates" in result and len(result["candidates"]) > 0:
                        text = result["candidates"][0]["content"]["parts"][0]["text"]
                        
                        # Generate contextual suggestions
                        suggestions = await get_contextual_suggestions(message, level)
                        
                        logger.info("Gemini response generated successfully via REST API")
                        return ChatResponse(
                            response=text.strip(),
                            suggestions=suggestions[:4]
                        )
                    else:
                        logger.warning("No candidates in Gemini response")
                        return get_offline_response(message, level)
                else:
                    error_text = await response.text()
                    logger.error(f"Gemini REST API error {response.status}: {error_text}")
                    return get_offline_response(message, level)
                    
    except Exception as e:
        logger.error(f"Gemini REST API error: {e}")
        return get_offline_response(message, level)

async def get_contextual_suggestions(message: str, level: str) -> List[str]:
    """
    Generate contextual follow-up suggestions based on the user's question
    """
    # Smart suggestions based on keywords in the message
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["what", "exoplanet", "planet"]):
        return [
            "How do we detect exoplanets?",
            "What is the transit method?", 
            "Tell me about habitable zones",
            "What's the most Earth-like exoplanet?"
        ]
    elif any(word in message_lower for word in ["detect", "find", "discovery"]):
        return [
            "What is the Kepler mission?",
            "How does TESS work?",
            "What is gravitational microlensing?",
            "Tell me about radial velocity method"
        ]
    elif any(word in message_lower for word in ["kepler", "tess", "mission"]):
        return [
            "What are the most famous exoplanet discoveries?",
            "How many exoplanets have been found?",
            "What's next in exoplanet research?",
            "Tell me about the James Webb telescope"
        ]
    elif any(word in message_lower for word in ["life", "habitable", "water"]):
        return [
            "What makes a planet habitable?",
            "Have we found signs of life?",
            "What is the Goldilocks zone?",
            "Tell me about biosignatures"
        ]
    else:
        # Default suggestions based on level
        return SUGGESTIONS.get(level, SUGGESTIONS["beginner"])[:4]

# API Endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat_with_assistant(message: ChatMessage):
    """
    Chat with the NASA Exoplanet Assistant powered by Gemini AI
    """
    try:
        logger.info(f"Chat request: {message.message[:50]}... | Level: {message.level}")
        
        # Use Gemini AI for intelligent responses
        if GEMINI_AVAILABLE:
            response = await get_gemini_response(message.message, message.level)
            return response
        else:
            logger.warning("Gemini not available, using fallback response")
            return get_offline_response(message.message, message.level)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Fallback to offline response with explicit parameters
        try:
            return get_offline_response(message.message, message.level)
        except Exception as fallback_error:
            logger.error(f"Fallback error: {fallback_error}")
            # Return a basic response if everything fails
            return ChatResponse(
                response="I apologize, but I'm having technical difficulties. Please try asking about exoplanets, detection methods, or NASA missions!",
                suggestions=["What are exoplanets?", "How do we detect exoplanets?", "Tell me about Kepler mission", "What is TESS?"]
            )

@router.get("/suggestions", response_model=List[str])
async def get_suggestions(level: str = "beginner"):
    """
    Get suggested questions based on user level
    """
    try:
        suggestions = SUGGESTIONS.get(level, SUGGESTIONS["beginner"])
        return suggestions
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return SUGGESTIONS["beginner"]

@router.post("/set-level")
async def set_user_level(level_data: UserLevel, user_id: str = "anonymous"):
    """
    Set user learning level (beginner, intermediate, advanced)
    """
    try:
        valid_levels = ["beginner", "intermediate", "advanced"]
        if level_data.level not in valid_levels:
            raise HTTPException(status_code=400, detail="Invalid level. Use: beginner, intermediate, or advanced")
        
        # In a real app, you'd save this to a database
        logger.info(f"User {user_id} set level to: {level_data.level}")
        
        return {"message": f"Learning level set to {level_data.level}", "level": level_data.level}
        
    except Exception as e:
        logger.error(f"Set level error: {e}")
        raise HTTPException(status_code=500, detail="Error setting user level")

@router.post("/level")
async def update_user_level(level_data: UserLevel):
    """
    Update user learning level (alternative endpoint for frontend compatibility)
    """
    try:
        valid_levels = ["beginner", "intermediate", "advanced"]
        if level_data.level not in valid_levels:
            raise HTTPException(status_code=400, detail="Invalid level. Use: beginner, intermediate, or advanced")
        
        logger.info(f"User level updated to: {level_data.level}")
        return {"message": f"Level updated to {level_data.level}", "level": level_data.level}
        
    except Exception as e:
        logger.error(f"Error updating level: {e}")
        raise HTTPException(status_code=500, detail="Failed to update level")

@router.get("/health")
async def chatbot_health():
    """Health check for chatbot service"""
    return {
        "status": "healthy",
        "service": "NASA Exoplanet Chatbot",
        "version": "1.0.0",
        "features": ["educational_chat", "level_based_learning", "exoplanet_knowledge"]
    }