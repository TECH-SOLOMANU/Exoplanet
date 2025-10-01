import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Card,
  CardContent,
  Grid,
  Chip,
  Box,
  CircularProgress,
  Alert
} from '@mui/material';
import axios from 'axios';

const Models = () => {
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const response = await axios.get('http://localhost:800/api/v1/predictions/model-status');
      setModelStatus(response.data);
    } catch (err) {
      console.error('Model status error:', err);
      setError('Failed to fetch model status');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, textAlign: 'center' }}>
        <CircularProgress />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        ML Models Status
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      {modelStatus && (
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                System Status
              </Typography>
              <Chip 
                label={modelStatus.ready_for_predictions ? 'Ready' : 'Not Ready'}
                color={modelStatus.ready_for_predictions ? 'success' : 'error'}
              />
            </Paper>
          </Grid>

          {modelStatus.models?.tabular_model && (
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Tabular Model
                  </Typography>
                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2">
                      Status: 
                      <Chip 
                        size="small"
                        label={modelStatus.models.tabular_model.loaded ? 'Loaded' : 'Not Loaded'}
                        color={modelStatus.models.tabular_model.loaded ? 'success' : 'error'}
                        sx={{ ml: 1 }}
                      />
                    </Typography>
                  </Box>
                  <Typography variant="body2" gutterBottom>
                    Type: {modelStatus.models.tabular_model.type}
                  </Typography>
                  {modelStatus.models.tabular_model.features && (
                    <Box>
                      <Typography variant="body2" gutterBottom>
                        Features ({modelStatus.models.tabular_model.features.length}):
                      </Typography>
                      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                        {modelStatus.models.tabular_model.features.map((feature, index) => (
                          <Chip key={index} label={feature} size="small" variant="outlined" />
                        ))}
                      </Box>
                    </Box>
                  )}
                </CardContent>
              </Card>
            </Grid>
          )}

          {modelStatus.models?.scaler && (
            <Grid item xs={12} md={6}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Data Preprocessing
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Scaler Status: 
                    <Chip 
                      size="small"
                      label={modelStatus.models.scaler.loaded ? 'Loaded' : 'Not Loaded'}
                      color={modelStatus.models.scaler.loaded ? 'success' : 'error'}
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                  <Typography variant="body2">
                    Type: {modelStatus.models.scaler.type}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {modelStatus.models?.light_curve_model && (
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Light Curve Analysis
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Status: 
                    <Chip 
                      size="small"
                      label={modelStatus.models.light_curve_model.loaded ? 'Available' : 'Not Available'}
                      color={modelStatus.models.light_curve_model.loaded ? 'success' : 'error'}
                      sx={{ ml: 1 }}
                    />
                  </Typography>
                  <Typography variant="body2" gutterBottom>
                    Type: {modelStatus.models.light_curve_model.type}
                  </Typography>
                  <Typography variant="body2">
                    Method: {modelStatus.models.light_curve_model.method}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      )}
    </Container>
  );
};

export default Models;