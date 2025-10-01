import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Button,
  Grid,
  Card,
  CardContent,
  TextField,
  Box,
  Alert,
  CircularProgress,
  Chip
} from '@mui/material';
import axios from 'axios';

const Predictions = () => {
  const [formData, setFormData] = useState({
    orbital_period: '',
    planet_radius: '',
    planet_mass: '',
    equilibrium_temp: '',
    stellar_temp: '',
    stellar_radius: '',
    stellar_mass: ''
  });
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleInputChange = (event) => {
    const { name, value } = event.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError('');
    setPrediction(null);

    try {
      const response = await axios.post('http://localhost:800/api/v1/predictions/predict', formData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const fillExampleData = (example) => {
    const examples = {
      earth: {
        orbital_period: 365.25,
        planet_radius: 1.0,
        planet_mass: 1.0,
        equilibrium_temp: 288,
        stellar_temp: 5778,
        stellar_radius: 1.0,
        stellar_mass: 1.0
      },
      hotJupiter: {
        orbital_period: 3.5,
        planet_radius: 1.2,
        planet_mass: 0.8,
        equilibrium_temp: 1200,
        stellar_temp: 6000,
        stellar_radius: 1.1,
        stellar_mass: 1.05
      },
      superEarth: {
        orbital_period: 22.7,
        planet_radius: 1.6,
        planet_mass: 4.8,
        equilibrium_temp: 450,
        stellar_temp: 5200,
        stellar_radius: 0.9,
        stellar_mass: 0.85
      }
    };
    setFormData(examples[example]);
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Exoplanet Classification Predictions
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Planet Parameters
            </Typography>
            
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Quick Examples:
              </Typography>
              <Button 
                size="small" 
                onClick={() => fillExampleData('earth')}
                sx={{ mr: 1 }}
              >
                Earth-like
              </Button>
              <Button 
                size="small" 
                onClick={() => fillExampleData('hotJupiter')}
                sx={{ mr: 1 }}
              >
                Hot Jupiter
              </Button>
              <Button 
                size="small" 
                onClick={() => fillExampleData('superEarth')}
              >
                Super Earth
              </Button>
            </Box>

            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Orbital Period (days)"
                  name="orbital_period"
                  type="number"
                  value={formData.orbital_period}
                  onChange={handleInputChange}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Planet Radius (Earth radii)"
                  name="planet_radius"
                  type="number"
                  value={formData.planet_radius}
                  onChange={handleInputChange}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Planet Mass (Earth masses)"
                  name="planet_mass"
                  type="number"
                  value={formData.planet_mass}
                  onChange={handleInputChange}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Equilibrium Temperature (K)"
                  name="equilibrium_temp"
                  type="number"
                  value={formData.equilibrium_temp}
                  onChange={handleInputChange}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Stellar Temperature (K)"
                  name="stellar_temp"
                  type="number"
                  value={formData.stellar_temp}
                  onChange={handleInputChange}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  fullWidth
                  label="Stellar Radius (Solar radii)"
                  name="stellar_radius"
                  type="number"
                  value={formData.stellar_radius}
                  onChange={handleInputChange}
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Stellar Mass (Solar masses)"
                  name="stellar_mass"
                  type="number"
                  value={formData.stellar_mass}
                  onChange={handleInputChange}
                />
              </Grid>
            </Grid>

            <Button
              variant="contained"
              onClick={handlePredict}
              disabled={loading}
              sx={{ mt: 2 }}
              fullWidth
            >
              {loading ? <CircularProgress size={24} /> : 'Predict Classification'}
            </Button>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Prediction Results
            </Typography>
            
            {error && (
              <Alert severity="error" sx={{ mb: 2 }}>
                {error}
              </Alert>
            )}

            {prediction && (
              <Box>
                <Card sx={{ mb: 2 }}>
                  <CardContent>
                    <Typography variant="h5" gutterBottom>
                      Classification: 
                      <Chip 
                        label={prediction.prediction} 
                        color={
                          prediction.prediction === 'CONFIRMED' ? 'success' :
                          prediction.prediction === 'CANDIDATE' ? 'warning' : 'error'
                        }
                        sx={{ ml: 1 }}
                      />
                    </Typography>
                    <Typography variant="h6" color="text.secondary">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </Typography>
                  </CardContent>
                </Card>

                {prediction.probabilities && (
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Class Probabilities
                      </Typography>
                      {Object.entries(prediction.probabilities).map(([className, prob]) => (
                        <Box key={className} sx={{ mb: 1 }}>
                          <Typography variant="body2">
                            {className}: {(prob * 100).toFixed(1)}%
                          </Typography>
                          <Box
                            sx={{
                              width: '100%',
                              height: 8,
                              backgroundColor: 'grey.300',
                              borderRadius: 1,
                              overflow: 'hidden'
                            }}
                          >
                            <Box
                              sx={{
                                width: `${prob * 100}%`,
                                height: '100%',
                                backgroundColor: 
                                  className === 'CONFIRMED' ? 'success.main' :
                                  className === 'CANDIDATE' ? 'warning.main' : 'error.main'
                              }}
                            />
                          </Box>
                        </Box>
                      ))}
                    </CardContent>
                  </Card>
                )}

                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Recommendation
                    </Typography>
                    <Typography variant="body1">
                      {prediction.recommendation}
                    </Typography>
                  </CardContent>
                </Card>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Predictions;