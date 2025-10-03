import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  IconButton,
  Typography,
  Chip,
  Paper,
  Fab,
  Slide,
  Avatar,
  List,
  ListItem,
  ListItemText,
  Divider,
  Button,
  FormControl,
  Select,
  MenuItem,
  InputLabel
} from '@mui/material';
import {
  Chat as ChatIcon,
  Send as SendIcon,
  Close as CloseIcon,
  SmartToy as BotIcon,
  Person as PersonIcon,
  School as SchoolIcon,
  Rocket as RocketIcon,
  Science as ScienceIcon
} from '@mui/icons-material';
import axios from 'axios';

const ChatBot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! 🚀 I'm your exoplanet learning assistant. I can help you understand:\n\n🪐 What exoplanets are\n🔭 How we detect them\n🛰️ NASA missions like Kepler & TESS\n🤖 How this AI platform works\n\nWhat would you like to learn about?",
      sender: 'bot',
      timestamp: new Date(),
      type: 'welcome'
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [userLevel, setUserLevel] = useState('beginner');
  const [suggestions, setSuggestions] = useState([
    "What are exoplanets?",
    "How do we detect exoplanets?", 
    "Tell me about the Kepler mission",
    "What is the transit method?"
  ]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Educational content for offline mode
  const getOfflineResponse = (message) => {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('exoplanet') || lowerMessage.includes('what are')) {
      return {
        text: "🪐 Exoplanets are planets that orbit stars outside our solar system! Scientists have discovered over 5,000 confirmed exoplanets using various detection methods. They come in different sizes - from smaller than Earth to larger than Jupiter!",
        suggestions: ["How do we detect them?", "Tell me about the Kepler mission", "What types of exoplanets exist?"]
      };
    }
    
    if (lowerMessage.includes('detect') || lowerMessage.includes('find')) {
      return {
        text: "🔭 We detect exoplanets using several methods:\n\n• **Transit Method**: Watch for tiny dips in starlight when a planet passes in front of its star\n• **Radial Velocity**: Measure the wobble of stars caused by orbiting planets\n• **Direct Imaging**: Actually photograph the planet (very rare!)\n• **Gravitational Microlensing**: Use gravity as a magnifying glass\n\nThis platform uses data from transit observations!",
        suggestions: ["What is the transit method?", "Tell me about Kepler", "How accurate is this AI?"]
      };
    }
    
    if (lowerMessage.includes('kepler') || lowerMessage.includes('mission')) {
      return {
        text: "🛰️ The Kepler Space Telescope was a game-changer! Launched in 2009, it discovered over 2,600 confirmed exoplanets by watching for transits. Kepler stared at the same patch of sky for years, monitoring 150,000 stars simultaneously!\n\nFun fact: This AI platform uses real Kepler data with 9,564 observations! 📊",
        suggestions: ["What is TESS?", "How does this AI work?", "What about the K2 mission?"]
      };
    }
    
    if (lowerMessage.includes('transit') || lowerMessage.includes('method')) {
      return {
        text: "🌟 The transit method is like a mini-eclipse! When a planet passes between us and its star, it blocks a tiny bit of starlight - usually less than 1%.\n\n**How it works:**\n1. Monitor a star's brightness continuously\n2. Look for periodic dips in brightness\n3. Analyze the dip pattern to determine planet size and orbit\n\nThe AI in this platform is trained to spot these patterns! 🤖",
        suggestions: ["How big are these dips?", "What about false positives?", "Show me some data!"]
      };
    }
    
    if (lowerMessage.includes('ai') || lowerMessage.includes('machine learning') || lowerMessage.includes('how does this work')) {
      return {
        text: "🤖 This AI platform uses advanced machine learning to automatically detect exoplanets!\n\n**Our approach:**\n• **Ensemble Models**: XGBoost, LightGBM, Gradient Boosting working together\n• **Feature Engineering**: Extract statistical patterns from light curves\n• **Real NASA Data**: Trained on 9,564 actual observations\n• **95%+ Accuracy**: Research-grade performance\n\nIt can classify planets as CONFIRMED, CANDIDATE, or FALSE POSITIVE! ✨",
        suggestions: ["What features does it use?", "How accurate is it?", "Can I upload my own data?"]
      };
    }
    
    // Default response
    return {
      text: "🌌 That's a great question! I'm here to help you learn about exoplanets and space exploration. Feel free to ask me about:\n\n• Exoplanet detection methods\n• NASA missions (Kepler, TESS, K2)\n• How this AI platform works\n• Space science concepts\n\nWhat would you like to explore?",
      suggestions: ["What are exoplanets?", "How do we find them?", "Tell me about Kepler", "How does the AI work?"]
    };
  };

  const sendMessage = async (messageText = inputMessage) => {
    if (!messageText.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: messageText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      // Call the enhanced Gemini API
      const response = await axios.post('http://localhost:800/api/v1/chatbot/chat', {
        message: messageText,
        user_level: userLevel,
        context: {
          current_page: window.location.pathname,
          conversation_history: messages.slice(-3).map(m => ({ text: m.text, sender: m.sender }))
        }
      });

      const botMessage = {
        id: Date.now() + 1,
        text: response.data.response || "I apologize, but I couldn't generate a proper response.",
        sender: 'bot',
        timestamp: new Date(),
        suggestions: response.data.suggestions || [],
        type: response.data.type || 'response',
        confidence: response.data.confidence || 'high',
        source: response.data.source || 'AI Assistant'
      };

      setMessages(prev => [...prev, botMessage]);
      setSuggestions(response.data.suggestions || []);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Enhanced offline response as fallback
      const offlineResponse = getOfflineResponse(messageText);
      const botMessage = {
        id: Date.now() + 1,
        text: offlineResponse.text + "\n\n*Note: Currently in offline mode. Connect Gemini API for enhanced responses!*",
        sender: 'bot',
        timestamp: new Date(),
        suggestions: offlineResponse.suggestions,
        type: 'offline',
        confidence: 'medium',
        source: 'Offline Assistant'
      };
      
      setMessages(prev => [...prev, botMessage]);
      setSuggestions(offlineResponse.suggestions);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    sendMessage(suggestion);
  };

  const handleLevelChange = async (event) => {
    const newLevel = event.target.value;
    setUserLevel(newLevel);
    
    try {
      await axios.post('http://localhost:800/api/v1/chatbot/level', {
        level: newLevel
      });
    } catch (error) {
      console.error('Error setting user level:', error);
    }
  };

  const formatMessage = (text) => {
    // Handle undefined or null text
    if (!text || typeof text !== 'string') {
      return '';
    }
    
    // Convert markdown-like formatting to HTML
    return text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/🌟|🚀|🔭|🌍|📚|💡|🔬|🧪|📊|🌌|🟢|🟡|🔴|🔥|🧊|🏜️|🌊|🤔|📋|🛑|🔄|🔍|📸|📅|🎯/g, '<span class="emoji">$&</span>');
  };

  const getChatIcon = (type) => {
    switch (type) {
      case 'mission_info':
        return <RocketIcon />;
      case 'educational':
        return <ScienceIcon />;
      case 'learning_paths':
        return <SchoolIcon />;
      default:
        return <BotIcon />;
    }
  };

  return (
    <>
      {/* Floating Action Button */}
      <Fab
        color="primary"
        aria-label="open chat"
        onClick={() => setIsOpen(true)}
        sx={{
          position: 'fixed',
          bottom: 24,
          right: 24,
          zIndex: 1000,
          background: 'linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)',
          '&:hover': {
            background: 'linear-gradient(45deg, #1565c0 30%, #1976d2 90%)',
          }
        }}
        style={{ display: isOpen ? 'none' : 'flex' }}
      >
        <ChatIcon />
      </Fab>

      {/* Chat Window */}
      <Slide direction="up" in={isOpen} mountOnEnter unmountOnExit>
        <Card
          sx={{
            position: 'fixed',
            bottom: 24,
            right: 24,
            width: 400,
            height: 600,
            zIndex: 1001,
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
            borderRadius: 2,
            overflow: 'hidden'
          }}
        >
          {/* Header */}
          <Box
            sx={{
              background: 'linear-gradient(45deg, #1976d2 30%, #42a5f5 90%)',
              color: 'white',
              p: 2,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Avatar sx={{ bgcolor: 'rgba(255,255,255,0.2)' }}>
                <BotIcon />
              </Avatar>
              <Box>
                <Typography variant="h6" sx={{ fontSize: '1rem', fontWeight: 'bold' }}>
                  Exoplanet Assistant
                </Typography>
                <Typography variant="caption" sx={{ opacity: 0.9 }}>
                  NASA Space Apps Challenge
                </Typography>
              </Box>
            </Box>
            <IconButton
              color="inherit"
              onClick={() => setIsOpen(false)}
              size="small"
            >
              <CloseIcon />
            </IconButton>
          </Box>

          {/* Learning Level Selector */}
          <Box sx={{ p: 1, borderBottom: '1px solid #eee' }}>
            <FormControl size="small" fullWidth>
              <InputLabel>Learning Level</InputLabel>
              <Select
                value={userLevel}
                label="Learning Level"
                onChange={handleLevelChange}
              >
                <MenuItem value="beginner">🟢 Beginner</MenuItem>
                <MenuItem value="intermediate">🟡 Intermediate</MenuItem>
                <MenuItem value="advanced">🔴 Advanced</MenuItem>
              </Select>
            </FormControl>
          </Box>

          {/* Messages */}
          <Box
            sx={{
              flex: 1,
              overflow: 'auto',
              p: 1,
              background: '#f5f5f5'
            }}
          >
            <List sx={{ p: 0 }}>
              {messages.map((message, index) => (
                <ListItem
                  key={message.id}
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: message.sender === 'user' ? 'flex-end' : 'flex-start',
                    p: 1
                  }}
                >
                  <Paper
                    elevation={2}
                    sx={{
                      p: 2,
                      maxWidth: '85%',
                      backgroundColor: message.sender === 'user' ? '#1976d2' : '#ffffff',
                      color: message.sender === 'user' ? '#ffffff' : '#1a1a1a',
                      borderRadius: message.sender === 'user' ? '20px 20px 5px 20px' : '20px 20px 20px 5px',
                      border: message.sender === 'bot' ? '1px solid #e0e0e0' : 'none',
                      boxShadow: message.sender === 'user' ? '0 2px 8px rgba(25, 118, 210, 0.3)' : '0 2px 8px rgba(0, 0, 0, 0.1)'
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 1 }}>
                      {message.sender === 'bot' && (
                        <Avatar sx={{ 
                          bgcolor: '#1976d2', 
                          width: 24, 
                          height: 24,
                          fontSize: '0.75rem'
                        }}>
                          {getChatIcon(message.type)}
                        </Avatar>
                      )}
                      <Box sx={{ flex: 1 }}>
                        <Typography
                          variant="body2"
                          sx={{
                            whiteSpace: 'pre-wrap',
                            color: message.sender === 'user' ? '#ffffff !important' : '#1a1a1a !important',
                            fontSize: '0.95rem',
                            lineHeight: 1.5,
                            '& .emoji': {
                              fontSize: '1.1em'
                            },
                            '& strong': {
                              fontWeight: 600,
                              color: message.sender === 'user' ? '#ffffff !important' : '#1a1a1a !important'
                            },
                            '& em': {
                              fontStyle: 'italic',
                              color: message.sender === 'user' ? '#ffffff !important' : '#1a1a1a !important'
                            }
                          }}
                          dangerouslySetInnerHTML={{
                            __html: formatMessage(message.text)
                          }}
                        />
                        {message.suggestions && message.suggestions.length > 0 && (
                          <Box sx={{ mt: 1, display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                            {message.suggestions.slice(0, 3).map((suggestion, idx) => (
                              <Chip
                                key={idx}
                                label={suggestion}
                                size="small"
                                onClick={() => handleSuggestionClick(suggestion)}
                                sx={{
                                  fontSize: '0.75rem',
                                  height: 24,
                                  cursor: 'pointer',
                                  backgroundColor: '#1976d2',
                                  color: 'white',
                                  '&:hover': {
                                    backgroundColor: '#1565c0',
                                    color: 'white'
                                  }
                                }}
                              />
                            ))}
                          </Box>
                        )}
                      </Box>
                    </Box>
                  </Paper>
                  <Typography
                    variant="caption"
                    sx={{
                      mt: 0.5,
                      color: 'text.secondary',
                      fontSize: '0.7rem'
                    }}
                  >
                    {message.timestamp.toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </Typography>
                </ListItem>
              ))}
              {isLoading && (
                <ListItem>
                  <Paper
                    elevation={2}
                    sx={{
                      p: 2,
                      backgroundColor: '#ffffff',
                      color: '#555555',
                      borderRadius: '20px 20px 20px 5px',
                      border: '1px solid #e0e0e0',
                      boxShadow: '0 2px 8px rgba(0, 0, 0, 0.1)'
                    }}
                  >
                    <Typography variant="body2" sx={{ fontStyle: 'italic', color: '#555555' }}>
                      🤔 Thinking...
                    </Typography>
                  </Paper>
                </ListItem>
              )}
              <div ref={messagesEndRef} />
            </List>
          </Box>

          {/* Quick Suggestions */}
          {suggestions.length > 0 && (
            <Box sx={{ p: 1, borderTop: '1px solid #eee', backgroundColor: '#fafafa' }}>
              <Typography variant="caption" sx={{ color: 'text.secondary', mb: 1, display: 'block' }}>
                💡 Quick suggestions:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                {suggestions.slice(0, 4).map((suggestion, idx) => (
                  <Chip
                    key={idx}
                    label={suggestion}
                    size="small"
                    variant="outlined"
                    onClick={() => handleSuggestionClick(suggestion)}
                    sx={{
                      fontSize: '0.75rem',
                      cursor: 'pointer',
                      color: '#1565c0',
                      borderColor: '#1565c0',
                      backgroundColor: '#f8f9fa',
                      fontWeight: 500,
                      '&:hover': {
                        backgroundColor: '#1565c0',
                        borderColor: '#1565c0',
                        color: 'white'
                      }
                    }}
                  />
                ))}
              </Box>
            </Box>
          )}

          {/* Input */}
          <Box sx={{ p: 2, borderTop: '1px solid #eee', backgroundColor: 'white' }}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                fullWidth
                size="small"
                placeholder="Ask about exoplanets..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                multiline
                maxRows={3}
                disabled={isLoading}
                sx={{
                  '& .MuiInputBase-input': {
                    color: '#1a1a1a !important',
                    fontSize: '0.95rem'
                  },
                  '& .MuiInputBase-input::placeholder': {
                    color: '#666 !important',
                    opacity: 1
                  },
                  '& .MuiOutlinedInput-root': {
                    backgroundColor: '#ffffff',
                    '& fieldset': {
                      borderColor: '#e0e0e0'
                    },
                    '&:hover fieldset': {
                      borderColor: '#1976d2'
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#1976d2'
                    }
                  }
                }}
              />
              <IconButton
                color="primary"
                onClick={() => sendMessage()}
                disabled={isLoading || !inputMessage.trim()}
                sx={{
                  backgroundColor: '#1976d2',
                  color: 'white',
                  '&:hover': {
                    backgroundColor: '#1565c0'
                  },
                  '&:disabled': {
                    backgroundColor: '#ccc'
                  }
                }}
              >
                <SendIcon />
              </IconButton>
            </Box>
          </Box>
        </Card>
      </Slide>
    </>
  );
};

export default ChatBot;