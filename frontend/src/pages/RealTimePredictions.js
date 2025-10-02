import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  TextField,
  Button,
  Box,
  Alert,
  Chip,
  LinearProgress
} from '@mui/material';
import { Radar, TrendingUp, Science, CheckCircle } from '@mui/icons-material';
import Plot from 'react-plotly.js';

const RealTimePredictions = () => {
  const [inputData, setInputData] = useState({
    pl_orbper: '',      // Orbital Period (days)
    pl_rade: '',        // Planet Radius (Earth radii)
    pl_eqt: '',         // Equilibrium Temperature (K)
    koi_depth: '',      // Transit Depth (ppm)
    koi_duration: '',   // Transit Duration (hours)
    koi_impact: '',     // Impact Parameter
    st_teff: ''         // Stellar Temperature (K)
  });
  
  const [prediction, setPrediction] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [predictionHistory, setPredictionHistory] = useState([]);

  const handleInputChange = (field, value) => {
    setInputData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const analyzePlanet = async () => {
    setIsAnalyzing(true);
    try {
      const response = await fetch('http://localhost:800/api/v1/predictions/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(inputData)
      });
      
      if (response.ok) {
        const result = await response.json();
        setPrediction(result.prediction);
        setConfidence(result.confidence);
        
        // Add to history
        const newEntry = {
          timestamp: new Date().toLocaleTimeString(),
          prediction: result.prediction,
          confidence: result.confidence,
          data: { ...inputData }
        };
        setPredictionHistory(prev => [newEntry, ...prev.slice(0, 4)]);
      }
    } catch (error) {
      console.error('Prediction failed:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const getConfidenceColor = (conf) => {
    if (conf > 0.8) return 'success';
    if (conf > 0.6) return 'warning';
    return 'error';
  };

  const getPredictionIcon = (pred) => {
    switch(pred) {
      case 'CONFIRMED': return <CheckCircle color="success" />;
      case 'CANDIDATE': return <Science color="warning" />;
      case 'FALSE_POSITIVE': return <TrendingUp color="error" />;
      default: return <Radar />;
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <Radar sx={{ mr: 2, color: 'primary.main' }} />
        Real-Time Exoplanet Predictions
      </Typography>
      
      <Alert severity="info" sx={{ mb: 3 }}>
        <strong>AI-Powered Analysis:</strong> Enter planetary parameters to get instant 
        exoplanet classification using advanced machine learning models.
      </Alert>

      <Grid container spacing={3}>
        {/* Input Panel */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Planetary Parameters
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Orbital Period (days)"
                  type="number"
                  value={inputData.pl_orbper}
                  onChange={(e) => handleInputChange('pl_orbper', e.target.value)}
                  helperText="Time to complete one orbit"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Planet Radius (Earth radii)"
                  type="number"
                  value={inputData.pl_rade}
                  onChange={(e) => handleInputChange('pl_rade', e.target.value)}
                  helperText="Size relative to Earth"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Temperature (K)"
                  type="number"
                  value={inputData.pl_eqt}
                  onChange={(e) => handleInputChange('pl_eqt', e.target.value)}
                  helperText="Equilibrium temperature"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Transit Depth (ppm)"
                  type="number"
                  value={inputData.koi_depth}
                  onChange={(e) => handleInputChange('koi_depth', e.target.value)}
                  helperText="Brightness dip during transit"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Transit Duration (hours)"
                  type="number"
                  value={inputData.koi_duration}
                  onChange={(e) => handleInputChange('koi_duration', e.target.value)}
                  helperText="Time crossing star"
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  label="Impact Parameter"
                  type="number"
                  value={inputData.koi_impact}
                  onChange={(e) => handleInputChange('koi_impact', e.target.value)}
                  helperText="Orbital geometry (0-1)"
                />
              </Grid>
            </Grid>
            
            <Button
              variant="contained"
              fullWidth
              onClick={analyzePlanet}
              disabled={isAnalyzing}
              sx={{ mt: 3 }}
              startIcon={isAnalyzing ? null : <Science />}
            >
              {isAnalyzing ? 'Analyzing...' : 'Analyze Exoplanet'}
            </Button>
            
            {isAnalyzing && <LinearProgress sx={{ mt: 2 }} />}
          </Paper>
        </Grid>

        {/* Results Panel */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              AI Prediction Results
            </Typography>
            
            {prediction ? (
              <Box>
                <Card sx={{ mb: 2, bgcolor: prediction === 'CONFIRMED' ? 'success.light' : 
                                   prediction === 'CANDIDATE' ? 'warning.light' : 'error.light' }}>
                  <CardContent sx={{ display: 'flex', alignItems: 'center' }}>
                    {getPredictionIcon(prediction)}
                    <Box sx={{ ml: 2, flex: 1 }}>
                      <Typography variant="h5" component="div">
                        {prediction.replace('_', ' ')}
                      </Typography>
                      <Typography variant="body2">
                        Confidence: {(confidence * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                    <Chip 
                      label={`${(confidence * 100).toFixed(1)}%`}
                      color={getConfidenceColor(confidence)}
                    />
                  </CardContent>
                </Card>
                
                <Alert severity={
                  prediction === 'CONFIRMED' ? 'success' :
                  prediction === 'CANDIDATE' ? 'warning' : 'info'
                }>
                  <strong>
                    {prediction === 'CONFIRMED' && 'Confirmed Exoplanet: High confidence detection!'}
                    {prediction === 'CANDIDATE' && 'Planet Candidate: Requires further validation.'}
                    {prediction === 'FALSE_POSITIVE' && 'False Positive: Likely not a planet.'}
                  </strong>
                </Alert>
              </Box>
            ) : (
              <Typography variant="body1" color="text.secondary">
                Enter planetary parameters and click "Analyze" to get AI prediction results.
              </Typography>
            )}
          </Paper>
          
          {/* Prediction History */}
          {predictionHistory.length > 0 && (
            <Paper sx={{ p: 3, mt: 2 }}>
              <Typography variant="h6" gutterBottom>
                Recent Predictions
              </Typography>
              {predictionHistory.map((entry, index) => (
                <Box key={index} sx={{ mb: 1, p: 1, bgcolor: 'background.paper', borderRadius: 1 }}>
                  <Typography variant="body2">
                    {entry.timestamp}: <strong>{entry.prediction}</strong> ({(entry.confidence * 100).toFixed(1)}%)
                  </Typography>
                </Box>
              ))}
            </Paper>
          )}
        </Grid>
      </Grid>
    </Container>
  );
};

export default RealTimePredictions;