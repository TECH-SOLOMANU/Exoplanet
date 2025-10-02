import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  Card,
  CardContent,
  Grid,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import { Timeline, ShowChart } from '@mui/icons-material';
import Plot from 'react-plotly.js';

const LightCurveAnalysis = () => {
  const [lightCurveData, setLightCurveData] = useState(null);
  const [selectedPlanet, setSelectedPlanet] = useState('');
  const [transitDepth, setTransitDepth] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const samplePlanets = [
    'Kepler-1b',
    'TOI-715b', 
    'K2-18b',
    'TRAPPIST-1d'
  ];

  const generateLightCurve = async (planetName) => {
    setIsAnalyzing(true);
    try {
      // Simulate light curve analysis
      const response = await fetch(`http://localhost:800/api/v1/nasa/light-curve/${planetName}`);
      if (response.ok) {
        const data = await response.json();
        setLightCurveData(data);
        
        // Calculate transit depth
        const baseline = 1.0;
        const minFlux = Math.min(...data.flux);
        const depth = ((baseline - minFlux) / baseline) * 1000; // in ppm
        setTransitDepth(depth);
      }
    } catch (error) {
      console.error('Failed to fetch light curve:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderLightCurvePlot = () => {
    if (!lightCurveData) return null;

    return (
      <Plot
        data={[
          {
            x: lightCurveData.time,
            y: lightCurveData.flux,
            type: 'scatter',
            mode: 'lines',
            name: 'Stellar Flux',
            line: { color: '#1976d2', width: 1 }
          }
        ]}
        layout={{
          title: `Light Curve Analysis - ${selectedPlanet}`,
          xaxis: { 
            title: 'Time (days)',
            showgrid: true,
            gridcolor: '#f0f0f0'
          },
          yaxis: { 
            title: 'Normalized Flux',
            showgrid: true,
            gridcolor: '#f0f0f0'
          },
          plot_bgcolor: 'white',
          paper_bgcolor: 'white',
          font: { family: 'Roboto', size: 12 },
          showlegend: true,
          autosize: true
        }}
        style={{ width: '100%', height: '400px' }}
        config={{ displayModeBar: true, responsive: true }}
      />
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
          <ShowChart sx={{ mr: 2, color: 'primary.main' }} />
          Light Curve Analysis
        </Typography>
        
        <Alert severity="info" sx={{ mb: 3 }}>
          <strong>Transit Method:</strong> Analyze stellar brightness variations to detect exoplanets. 
          When a planet passes in front of its star, it causes a characteristic dip in brightness.
        </Alert>

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <Timeline sx={{ mr: 1 }} />
                  Select Exoplanet
                </Typography>
                
                <FormControl fullWidth sx={{ mb: 2 }}>
                  <InputLabel>Planet Name</InputLabel>
                  <Select
                    value={selectedPlanet}
                    label="Planet Name"
                    onChange={(e) => setSelectedPlanet(e.target.value)}
                  >
                    {samplePlanets.map((planet) => (
                      <MenuItem key={planet} value={planet}>
                        {planet}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <Button
                  variant="contained"
                  fullWidth
                  onClick={() => generateLightCurve(selectedPlanet)}
                  disabled={!selectedPlanet || isAnalyzing}
                  sx={{ mb: 2 }}
                >
                  {isAnalyzing ? 'Analyzing Light Curve...' : 'Generate Light Curve'}
                </Button>

                {transitDepth && (
                  <Alert severity="success">
                    <strong>Transit Detected!</strong><br />
                    Depth: {transitDepth.toFixed(2)} ppm<br />
                    This indicates a planetary transit event.
                  </Alert>
                )}
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Analysis Results
                </Typography>
                
                {lightCurveData ? (
                  <Box>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Mission:</strong> {lightCurveData.mission}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Data Points:</strong> {lightCurveData.time.length}
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 1 }}>
                      <strong>Observation Period:</strong> {Math.max(...lightCurveData.time).toFixed(1)} days
                    </Typography>
                    {transitDepth && (
                      <Typography variant="body2" color="success.main">
                        <strong>Transit Depth:</strong> {transitDepth.toFixed(2)} ppm
                      </Typography>
                    )}
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    Select a planet and generate its light curve to see analysis results.
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {lightCurveData && (
          <Box sx={{ mt: 3 }}>
            <Card>
              <CardContent>
                {renderLightCurvePlot()}
              </CardContent>
            </Card>
          </Box>
        )}
      </Paper>
    </Container>
  );
};

export default LightCurveAnalysis;