import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

// Components
import Navbar from './components/Navigation/Navbar';
import ChatBot from './components/ChatBot/ChatBot';
import Dashboard from './pages/Dashboard';
import Predictions from './pages/Predictions';
import Upload from './pages/Upload';
import ModelManagement from './pages/ModelManagement';
import LightCurveAnalysis from './pages/LightCurveAnalysis';
import About from './pages/About';

// NASA space theme
const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#f50057',
      light: '#ff5983',
      dark: '#c51162',
    },
    background: {
      default: '#0a0e27',
      paper: '#1a1a2e',
    },
    text: {
      primary: '#ffffff',
      secondary: '#b0b0b0',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 500,
    },
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          backgroundImage: 'linear-gradient(45deg, #1a1a2e 30%, #16213e 90%)',
          border: '1px solid #2c2c54',
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ 
          minHeight: '100vh',
          background: 'linear-gradient(135deg, #0a0e27 0%, #16213e 50%, #1a1a2e 100%)',
        }}>
          <Navbar />
          <Box component="main" sx={{ pt: 3, pb: 6 }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/upload" element={<Upload />} />
              <Route path="/models" element={<ModelManagement />} />
              <Route path="/light-curves" element={<LightCurveAnalysis />} />
              <Route path="/about" element={<About />} />
            </Routes>
          </Box>
          
          {/* Educational Chatbot - Available on all pages */}
          <ChatBot />
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;