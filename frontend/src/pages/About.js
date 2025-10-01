import React from 'react';
import {
  Container,
  Paper,
  Typography,
  Grid,
  Card,
  CardContent,
  Chip,
  Box
} from '@mui/material';

const About = () => {
  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        About This Project
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h5" gutterBottom>
              NASA Space Apps Challenge 2025
            </Typography>
            <Typography variant="h6" color="primary" gutterBottom>
              Fully Automated Exoplanet Detection Platform
            </Typography>
            <Typography variant="body1" paragraph>
              This platform implements a comprehensive exoplanet detection system using AI/ML models 
              trained on NASA's Kepler, K2, and TESS datasets. The system features automated data 
              ingestion, machine learning pipeline, and interactive web interface.
            </Typography>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Key Features
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Chip label="Automated NASA Data Fetching" color="primary" />
                <Chip label="Dual ML Models (Tabular + CNN)" color="primary" />
                <Chip label="Explainable AI with SHAP" color="primary" />
                <Chip label="Real-time Web Dashboard" color="primary" />
                <Chip label="MongoDB Storage" color="primary" />
                <Chip label="Docker Deployment" color="primary" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Technology Stack
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                <Typography variant="body2"><strong>Backend:</strong> FastAPI (Python)</Typography>
                <Typography variant="body2"><strong>Frontend:</strong> React.js + Material-UI</Typography>
                <Typography variant="body2"><strong>ML:</strong> TensorFlow, XGBoost, SHAP</Typography>
                <Typography variant="body2"><strong>Database:</strong> MongoDB</Typography>
                <Typography variant="body2"><strong>Visualization:</strong> Plotly.js</Typography>
                <Typography variant="body2"><strong>Deployment:</strong> Docker</Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Data Sources
              </Typography>
              <Typography variant="body1" paragraph>
                This platform integrates with multiple NASA data sources:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                <Chip label="NASA Exoplanet Archive" variant="outlined" />
                <Chip label="MAST (Mikulski Archive)" variant="outlined" />
                <Chip label="TESS Mission Data" variant="outlined" />
                <Chip label="Kepler Mission Data" variant="outlined" />
                <Chip label="K2 Mission Data" variant="outlined" />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                ML Model Architecture
              </Typography>
              <Typography variant="body1" paragraph>
                The system employs a dual-model approach for comprehensive exoplanet detection:
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    <strong>Tabular Model (XGBoost)</strong>
                  </Typography>
                  <Typography variant="body2">
                    Processes stellar and planetary parameters including orbital period, 
                    planet radius, mass, equilibrium temperature, and stellar characteristics.
                  </Typography>
                </Grid>
                <Grid item xs={12} md={6}>
                  <Typography variant="subtitle1" gutterBottom>
                    <strong>Light Curve Model (CNN)</strong>
                  </Typography>
                  <Typography variant="body2">
                    Analyzes time-series photometric data to detect transit signals 
                    and classify potential exoplanet candidates.
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default About;