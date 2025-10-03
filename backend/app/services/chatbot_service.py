"""
Educational Chatbot Service for Exoplanet Learning
NASA Space Apps Challenge 2025
"""

import json
import re
from typing import Dict, List, Optional
from datetime import datetime
import random

class ExoplanetChatbot:
    """AI-powered educational chatbot for exoplanet learning"""
    
    def __init__(self):
        self.knowledge_base = self._load_knowledge_base()
        self.conversation_history = []
        self.user_level = "beginner"  # beginner, intermediate, advanced
        
    def _load_knowledge_base(self) -> Dict:
        """Load comprehensive exoplanet knowledge base"""
        return {
            "definitions": {
                "exoplanet": "An exoplanet is a planet that orbits a star outside our solar system. Also called extrasolar planets, they come in many sizes and compositions.",
                "transit method": "A technique to detect exoplanets by measuring the tiny dimming of starlight when a planet passes in front of its host star.",
                "habitable zone": "The region around a star where liquid water could exist on a planet's surface, also called the 'Goldilocks zone'.",
                "kepler mission": "NASA's space telescope that discovered thousands of exoplanets using the transit method from 2009-2018.",
                "tess": "Transiting Exoplanet Survey Satellite - NASA's current exoplanet-hunting mission launched in 2018.",
                "hot jupiter": "A gas giant exoplanet that orbits very close to its star, resulting in extremely high temperatures.",
                "super earth": "An exoplanet with a mass larger than Earth's but smaller than Neptune's, typically 2-10 times Earth's mass.",
                "rocky planet": "A terrestrial planet composed primarily of rock and metal, similar to Earth, Venus, Mars, and Mercury.",
                "orbital period": "The time it takes for a planet to complete one orbit around its star.",
                "stellar classification": "A system that categorizes stars based on their temperature, with types O, B, A, F, G, K, M from hottest to coolest."
            },
            
            "facts": [
                "Over 5,000 exoplanets have been confirmed as of 2023!",
                "The first exoplanet was discovered in 1995 around the star 51 Pegasi.",
                "Some exoplanets rain diamonds due to extreme pressure and carbon-rich atmospheres.",
                "Kepler-452b is called 'Earth's cousin' because it's similar in size and orbits in its star's habitable zone.",
                "The closest exoplanet to Earth is Proxima Centauri b, about 4.2 light-years away.",
                "Some exoplanets are completely covered in deep oceans with no land masses.",
                "HD 189733b is a blue planet, but it's hot enough to rain molten glass!",
                "PSR B1257+12 b was the first confirmed exoplanet, discovered orbiting a pulsar in 1992.",
                "Some exoplanets have multiple suns, like the fictional planet Tatooine from Star Wars.",
                "The James Webb Space Telescope can analyze exoplanet atmospheres in unprecedented detail."
            ],
            
            "missions": {
                "kepler": {
                    "description": "Revolutionary space telescope that used transit photometry to discover thousands of exoplanets",
                    "timeline": "2009-2018",
                    "discoveries": "Over 2,600 confirmed exoplanets",
                    "method": "Transit photometry - detecting tiny dips in starlight"
                },
                "tess": {
                    "description": "Current NASA mission surveying the entire sky for exoplanets",
                    "timeline": "2018-present",
                    "discoveries": "Over 5,000 candidate exoplanets",
                    "method": "Transit photometry with wider field of view"
                },
                "k2": {
                    "description": "Extended Kepler mission after mechanical failures",
                    "timeline": "2014-2018",
                    "discoveries": "Over 400 confirmed exoplanets",
                    "method": "Modified transit photometry"
                },
                "jwst": {
                    "description": "James Webb Space Telescope - studying exoplanet atmospheres",
                    "timeline": "2021-present",
                    "discoveries": "Detailed atmospheric analysis",
                    "method": "Spectroscopy and direct imaging"
                }
            },
            
            "detection_methods": {
                "transit": "Measuring the dimming of starlight when a planet passes in front of its star",
                "radial_velocity": "Detecting the wobble of a star caused by a planet's gravitational pull",
                "direct_imaging": "Taking actual pictures of exoplanets (very challenging)",
                "gravitational_microlensing": "Using gravitational effects to magnify distant stars and reveal planets",
                "astrometry": "Measuring precise stellar positions to detect planetary influence"
            },
            
            "learning_paths": {
                "beginner": [
                    "What is an exoplanet?",
                    "How do we find exoplanets?",
                    "What makes a planet habitable?",
                    "Famous exoplanet discoveries",
                    "Different types of exoplanets"
                ],
                "intermediate": [
                    "Transit method mathematics",
                    "Stellar classifications and planet formation",
                    "Atmospheric analysis techniques",
                    "Statistical analysis of exoplanet populations",
                    "Biosignature detection"
                ],
                "advanced": [
                    "Machine learning in exoplanet detection",
                    "False positive analysis",
                    "Atmospheric modeling",
                    "Habitability indices",
                    "Future mission planning"
                ]
            }
        }
    
    def process_message(self, message: str, user_context: Optional[Dict] = None) -> Dict:
        """Process user message and generate educational response"""
        message_lower = message.lower().strip()
        
        # Store conversation
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "user_context": user_context or {}
        })
        
        # Analyze intent
        intent = self._analyze_intent(message_lower)
        response = self._generate_response(intent, message_lower)
        
        # Add response to history
        self.conversation_history[-1]["bot_response"] = response
        
        return response
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent from message"""
        
        # Greeting patterns
        if any(word in message for word in ["hello", "hi", "hey", "start", "help"]):
            return "greeting"
        
        # Definition requests
        if any(phrase in message for phrase in ["what is", "define", "explain", "meaning of"]):
            return "definition"
        
        # How-to questions
        if any(phrase in message for phrase in ["how do", "how to", "how are", "how can"]):
            return "how_to"
        
        # Facts and trivia
        if any(word in message for word in ["fact", "interesting", "cool", "amazing", "tell me"]):
            return "facts"
        
        # Mission information
        if any(word in message for word in ["kepler", "tess", "k2", "mission", "jwst", "telescope"]):
            return "missions"
        
        # Learning progression
        if any(word in message for word in ["learn", "study", "course", "lesson", "next"]):
            return "learning"
        
        # Specific topics
        if any(word in message for word in ["habitable", "goldilocks", "life"]):
            return "habitability"
        
        if any(word in message for word in ["transit", "detection", "find", "discover"]):
            return "detection"
        
        if any(word in message for word in ["types", "kinds", "categories"]):
            return "classification"
        
        # Default to general help
        return "general"
    
    def _generate_response(self, intent: str, message: str) -> Dict:
        """Generate educational response based on intent"""
        
        if intent == "greeting":
            return {
                "message": "🌟 Hello! I'm your exoplanet learning assistant! I'm here to help you discover the fascinating world of planets beyond our solar system.\n\n🚀 I can help you with:\n• Exoplanet basics and definitions\n• Detection methods and missions\n• Fun facts and discoveries\n• Learning paths for different levels\n\nWhat would you like to learn about first?",
                "type": "greeting",
                "suggestions": ["What is an exoplanet?", "How do we find exoplanets?", "Tell me a fun fact!", "Show me learning paths"],
                "level": "beginner"
            }
        
        elif intent == "definition":
            return self._handle_definition_request(message)
        
        elif intent == "how_to":
            return self._handle_how_to_request(message)
        
        elif intent == "facts":
            return self._handle_facts_request()
        
        elif intent == "missions":
            return self._handle_mission_request(message)
        
        elif intent == "learning":
            return self._handle_learning_request()
        
        elif intent == "habitability":
            return self._handle_habitability_request()
        
        elif intent == "detection":
            return self._handle_detection_request()
        
        elif intent == "classification":
            return self._handle_classification_request()
        
        else:
            return self._handle_general_request(message)
    
    def _handle_definition_request(self, message: str) -> Dict:
        """Handle definition requests"""
        
        # Find matching terms
        matches = []
        for term, definition in self.knowledge_base["definitions"].items():
            if term in message or any(word in message for word in term.split()):
                matches.append((term, definition))
        
        if matches:
            term, definition = matches[0]  # Take the first match
            return {
                "message": f"📚 **{term.title()}**\n\n{definition}\n\n💡 Would you like to learn more about related topics?",
                "type": "definition",
                "term": term,
                "suggestions": self._get_related_suggestions(term),
                "level": "beginner"
            }
        else:
            return {
                "message": "🤔 I'd love to help define that! Could you be more specific? Here are some topics I know about:\n\n" + 
                          "\n".join([f"• {term.title()}" for term in list(self.knowledge_base["definitions"].keys())[:5]]),
                "type": "help",
                "suggestions": list(self.knowledge_base["definitions"].keys())[:4],
                "level": "beginner"
            }
    
    def _handle_how_to_request(self, message: str) -> Dict:
        """Handle how-to questions"""
        
        if "find" in message or "detect" in message:
            return {
                "message": "🔭 **How We Find Exoplanets**\n\nThere are several amazing methods:\n\n🌟 **Transit Method** (most common)\n• Watch for tiny dips in starlight when a planet passes in front of its star\n• Like a mini eclipse that repeats regularly\n\n📊 **Radial Velocity**\n• Detect the star's wobble caused by a planet's gravity\n• Measures changes in star's motion toward/away from us\n\n📸 **Direct Imaging**\n• Actually taking pictures of exoplanets (super challenging!)\n• Requires blocking out the bright star's light\n\nWant to explore any of these methods in detail?",
                "type": "educational",
                "suggestions": ["Transit method details", "Radial velocity explained", "Direct imaging challenges", "Show me real examples"],
                "level": "intermediate"
            }
        
        elif "life" in message or "habitable" in message:
            return {
                "message": "🌍 **How We Search for Habitable Worlds**\n\n🔍 We look for:\n• **Right distance from star** (Habitable Zone)\n• **Liquid water possibility**\n• **Suitable atmosphere**\n• **Rocky composition**\n\n🧪 **Biosignatures** we search for:\n• Oxygen and water vapor\n• Methane and other gases\n• Temperature patterns\n\n🌟 The James Webb Space Telescope is revolutionizing our ability to study exoplanet atmospheres!",
                "type": "educational",
                "suggestions": ["What is the habitable zone?", "Biosignature detection", "JWST discoveries", "Earth-like exoplanets"],
                "level": "intermediate"
            }
        
        else:
            return {
                "message": "🤔 I'd love to help! Could you be more specific about what you'd like to learn how to do? Here are some popular topics:\n\n• How we find exoplanets\n• How we determine if a planet is habitable\n• How transit detection works\n• How we analyze exoplanet atmospheres",
                "type": "help",
                "suggestions": ["How do we find exoplanets?", "How do we find life?", "How does transit work?", "How do we study atmospheres?"],
                "level": "beginner"
            }
    
    def _handle_facts_request(self) -> Dict:
        """Handle fun facts requests"""
        
        fact = random.choice(self.knowledge_base["facts"])
        
        return {
            "message": f"🌟 **Amazing Exoplanet Fact!**\n\n{fact}\n\n🚀 Want to learn more about this topic or hear another fact?",
            "type": "fact",
            "suggestions": ["Another fact!", "Learn more about this", "Kepler discoveries", "TESS mission"],
            "level": "beginner"
        }
    
    def _handle_mission_request(self, message: str) -> Dict:
        """Handle mission-related requests"""
        
        # Find matching mission
        mission_match = None
        for mission_name in self.knowledge_base["missions"].keys():
            if mission_name in message:
                mission_match = mission_name
                break
        
        if mission_match:
            mission = self.knowledge_base["missions"][mission_match]
            return {
                "message": f"🚀 **{mission_match.upper()} Mission**\n\n{mission['description']}\n\n📅 **Timeline:** {mission['timeline']}\n🔍 **Method:** {mission['method']}\n🌟 **Discoveries:** {mission['discoveries']}\n\nWant to learn about other missions or dive deeper into this one?",
                "type": "mission_info",
                "mission": mission_match,
                "suggestions": ["Other missions", "Detection methods", "Mission discoveries", "How transit works"],
                "level": "intermediate"
            }
        else:
            missions_list = "\n".join([f"🚀 **{name.upper()}**: {info['description'][:60]}..." 
                                     for name, info in self.knowledge_base["missions"].items()])
            return {
                "message": f"🌌 **Exoplanet Missions**\n\nHere are the major exoplanet-hunting missions:\n\n{missions_list}\n\nWhich mission interests you most?",
                "type": "mission_overview",
                "suggestions": list(self.knowledge_base["missions"].keys()),
                "level": "beginner"
            }
    
    def _handle_learning_request(self) -> Dict:
        """Handle learning path requests"""
        
        paths = self.knowledge_base["learning_paths"]
        
        return {
            "message": "📚 **Exoplanet Learning Paths**\n\nChoose your level:\n\n🟢 **Beginner**: Perfect if you're new to exoplanets\n🟡 **Intermediate**: Good astronomy background\n🔴 **Advanced**: Ready for technical details\n\nEach path has curated topics to build your knowledge step by step!",
            "type": "learning_paths",
            "paths": paths,
            "suggestions": ["Beginner path", "Intermediate path", "Advanced path", "Custom learning"],
            "level": "meta"
        }
    
    def _handle_habitability_request(self) -> Dict:
        """Handle habitability questions"""
        
        return {
            "message": "🌍 **What Makes a Planet Habitable?**\n\n🌟 **The Habitable Zone**\n• Also called the 'Goldilocks Zone'\n• Not too hot, not too cold - just right for liquid water\n• Distance varies based on star's temperature\n\n🧪 **Other Factors:**\n• Rocky composition (like Earth)\n• Protective atmosphere\n• Magnetic field to shield from radiation\n• Stable orbit\n\n🔬 **Biosignatures** we look for:\n• Water vapor, oxygen, methane\n• Seasonal changes\n• Chemical disequilibrium",
            "type": "educational",
            "suggestions": ["Goldilocks zone examples", "Biosignature detection", "Earth-like planets", "Atmospheric analysis"],
            "level": "intermediate"
        }
    
    def _handle_detection_request(self) -> Dict:
        """Handle detection method questions"""
        
        methods = self.knowledge_base["detection_methods"]
        method_text = "\n\n".join([f"🔬 **{name.replace('_', ' ').title()}**\n{desc}" 
                                  for name, desc in methods.items()])
        
        return {
            "message": f"🔭 **Exoplanet Detection Methods**\n\n{method_text}\n\n💡 The transit method finds about 70% of all known exoplanets!",
            "type": "educational",
            "suggestions": ["Transit method details", "Radial velocity", "Direct imaging", "Real examples"],
            "level": "intermediate"
        }
    
    def _handle_classification_request(self) -> Dict:
        """Handle exoplanet classification questions"""
        
        return {
            "message": "🌍 **Types of Exoplanets**\n\n🔥 **Hot Jupiters**\n• Gas giants orbiting very close to their stars\n• Extremely hot (>1000°C)\n\n🌍 **Super-Earths**\n• Rocky planets 2-10 times Earth's mass\n• Most common type discovered\n\n🧊 **Ice Giants**\n• Like Neptune, mostly water/methane ice\n• Often found in outer regions\n\n🏜️ **Desert Worlds**\n• Rocky planets with thin/no atmospheres\n• Often tidally locked\n\n🌊 **Ocean Worlds**\n• Completely covered in deep oceans\n• May harbor life beneath the surface",
            "type": "educational",
            "suggestions": ["Hot Jupiter examples", "Super-Earth candidates", "Ocean worlds", "Tidally locked planets"],
            "level": "intermediate"
        }
    
    def _handle_general_request(self, message: str) -> Dict:
        """Handle general or unclear requests"""
        
        return {
            "message": "🤔 I'm here to help you learn about exoplanets! I can assist with:\n\n📚 **Definitions** - What is an exoplanet? Transit method? Habitable zone?\n🔭 **Detection** - How do we find planets around other stars?\n🚀 **Missions** - Kepler, TESS, K2, and James Webb\n🌟 **Fun Facts** - Amazing discoveries and weird worlds\n📖 **Learning Paths** - Structured learning for your level\n\nWhat interests you most?",
            "type": "help",
            "suggestions": ["What is an exoplanet?", "How do we find them?", "Tell me a fun fact!", "Show learning paths"],
            "level": "beginner"
        }
    
    def _get_related_suggestions(self, term: str) -> List[str]:
        """Get related topic suggestions"""
        
        related_map = {
            "exoplanet": ["Transit method", "Habitable zone", "Kepler mission"],
            "transit method": ["Kepler mission", "Light curves", "Detection methods"],
            "habitable zone": ["Super-Earths", "Biosignatures", "Ocean worlds"],
            "kepler mission": ["TESS mission", "Transit method", "Planet discoveries"],
            "tess": ["Kepler comparison", "Current discoveries", "Future missions"]
        }
        
        return related_map.get(term, ["Fun facts", "Detection methods", "Learning paths", "Missions"])
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def set_user_level(self, level: str):
        """Set user learning level"""
        if level in ["beginner", "intermediate", "advanced"]:
            self.user_level = level