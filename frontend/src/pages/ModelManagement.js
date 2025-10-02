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
  Chip,
  Divider,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Switch,
  FormControlLabel,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import { 
  Psychology,
  TrendingUp,
  Science,
  Settings,
  Assessment,
  AutoFixHigh,
  School,
  Engineering
} from '@mui/icons-material';
import Plot from 'react-plotly.js';

const ModelManagement = () => {
  const [modelStats, setModelStats] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [userMode, setUserMode] = useState('novice'); // 'novice' or 'researcher'
  const [hyperparams, setHyperparams] = useState({
    learning_rate: 0.001,
    n_estimators: 100,
    max_depth: 6,
    batch_size: 32,
    epochs: 50
  });
  const [trainingHistory, setTrainingHistory] = useState(null);

  useEffect(() => {
    fetchModelStats();
  }, []);

  const fetchModelStats = async () => {
    try {
      // Fetch both status and performance metrics
      const [statusResponse, performanceResponse] = await Promise.all([
        fetch('http://localhost:800/api/v1/models/status'),
        fetch('http://localhost:800/api/v1/models/performance')
      ]);
      
      if (statusResponse.ok) {
        const status = await statusResponse.json();
        let combinedStats = { ...status };
        
        // Add performance metrics if available
        if (performanceResponse.ok) {
          const performance = await performanceResponse.json();
          combinedStats = { ...combinedStats, ...performance };
        }
        
        setModelStats(combinedStats);
      }
    } catch (error) {
      console.error('Failed to fetch model stats:', error);
    }
  };

  const handleRetrainModel = async () => {
    setIsTraining(true);
    try {
      const response = await fetch('http://localhost:800/api/v1/models/retrain', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          hyperparameters: hyperparams,
          use_latest_data: true
        })
      });

      if (response.ok) {
        const result = await response.json();
        setTrainingHistory(result.training_history);
        setTimeout(() => fetchModelStats(), 2000);
      }
    } catch (error) {
      console.error('Training failed:', error);
    } finally {
      setIsTraining(false);
    }
  };

  const handleHyperparamChange = (param, value) => {
    setHyperparams(prev => ({
      ...prev,
      [param]: value
    }));
  };

  const handleUpdateNASAData = async () => {
    try {
      const response = await fetch('http://localhost:800/api/v1/models/update-nasa-data', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Data update started: ${result.message}`);
        // Refresh model stats after a delay to show updated record count
        setTimeout(() => fetchModelStats(), 5000);
      }
    } catch (error) {
      console.error('NASA data update failed:', error);
      alert('Failed to start data update');
    }
  };

  const renderNoviceInterface = () => (
    <Box>
      <Card sx={{ mb: 3, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            <School sx={{ mr: 1, verticalAlign: 'middle' }} />
            Welcome to Exoplanet Discovery!
          </Typography>
          <Typography variant="body1">
            This tool helps you discover new planets outside our solar system using artificial intelligence. 
            Our AI has been trained on real data from NASA's Kepler, K2, and TESS space missions.
          </Typography>
        </CardContent>
      </Card>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📊 Current Model Performance
              </Typography>
              {modelStats && (
                <>
                  <Box sx={{ textAlign: 'center', mb: 2 }}>
                    <Typography variant="h3" color="success.main">
                      {(modelStats.accuracy * 100).toFixed(1)}%
                    </Typography>
                    <Typography variant="body2">Overall Accuracy</Typography>
                  </Box>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Training Data:</strong> {modelStats.training_samples?.toLocaleString()} samples
                  </Typography>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    <strong>Last Updated:</strong> {new Date(modelStats.last_trained).toLocaleDateString()}
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🚀 Quick Actions
              </Typography>
              <Button
                variant="outlined"
                fullWidth
                sx={{ mb: 2 }}
                startIcon={<TrendingUp />}
                onClick={handleUpdateNASAData}
              >
                Get Latest NASA Data
              </Button>
              <Button
                variant="contained"
                fullWidth
                sx={{ mb: 2 }}
                startIcon={<AutoFixHigh />}
                onClick={handleRetrainModel}
                disabled={isTraining}
              >
                {isTraining ? 'Training AI...' : 'Improve AI with Latest Data'}
              </Button>
              <Typography variant="body2" color="text.secondary">
                First get the latest NASA data, then retrain the AI model to improve accuracy with more comprehensive data.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );

  const renderResearcherInterface = () => (
    <Box>
      <Card sx={{ mb: 3, bgcolor: 'secondary.light', color: 'secondary.contrastText' }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            <Engineering sx={{ mr: 1, verticalAlign: 'middle' }} />
            Advanced Model Management
          </Typography>
          <Typography variant="body1">
            Fine-tune hyperparameters, analyze model performance, and manage training pipelines for optimal exoplanet detection.
          </Typography>
        </CardContent>
      </Card>

      {/* Model Statistics */}
      {modelStats && (
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Model Performance Metrics
            </Typography>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    <TableCell>Metric</TableCell>
                    <TableCell align="right">Value</TableCell>
                    <TableCell align="right">Class: Confirmed</TableCell>
                    <TableCell align="right">Class: Candidate</TableCell>
                    <TableCell align="right">Class: False Positive</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  <TableRow>
                    <TableCell>Accuracy</TableCell>
                    <TableCell align="right">{(modelStats.accuracy * 100).toFixed(2)}%</TableCell>
                    <TableCell align="right">-</TableCell>
                    <TableCell align="right">-</TableCell>
                    <TableCell align="right">-</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Precision</TableCell>
                    <TableCell align="right">{modelStats.precision ? (modelStats.precision * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.confirmed?.precision ? (modelStats.class_metrics.confirmed.precision * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.candidate?.precision ? (modelStats.class_metrics.candidate.precision * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.false_positive?.precision ? (modelStats.class_metrics.false_positive.precision * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>Recall</TableCell>
                    <TableCell align="right">{modelStats.recall ? (modelStats.recall * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.confirmed?.recall ? (modelStats.class_metrics.confirmed.recall * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.candidate?.recall ? (modelStats.class_metrics.candidate.recall * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.false_positive?.recall ? (modelStats.class_metrics.false_positive.recall * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell>F1-Score</TableCell>
                    <TableCell align="right">{modelStats.f1_score ? (modelStats.f1_score * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.confirmed?.f1_score ? (modelStats.class_metrics.confirmed.f1_score * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.candidate?.f1_score ? (modelStats.class_metrics.candidate.f1_score * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                    <TableCell align="right">{modelStats.class_metrics?.false_positive?.f1_score ? (modelStats.class_metrics.false_positive.f1_score * 100).toFixed(2) + '%' : 'N/A'}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}

      {/* Hyperparameter Tuning */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            <Settings sx={{ mr: 1, verticalAlign: 'middle' }} />
            Hyperparameter Tuning
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Learning Rate</Typography>
              <Slider
                value={hyperparams.learning_rate}
                min={0.0001}
                max={0.01}
                step={0.0001}
                onChange={(e, value) => handleHyperparamChange('learning_rate', value)}
                valueLabelDisplay="auto"
                valueLabelFormat={(value) => value.toFixed(4)}
              />
              <Typography variant="body2" color="text.secondary">
                Current: {hyperparams.learning_rate.toFixed(4)}
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Number of Estimators (XGBoost)</Typography>
              <Slider
                value={hyperparams.n_estimators}
                min={50}
                max={500}
                step={10}
                onChange={(e, value) => handleHyperparamChange('n_estimators', value)}
                valueLabelDisplay="auto"
              />
              <Typography variant="body2" color="text.secondary">
                Current: {hyperparams.n_estimators}
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Max Depth</Typography>
              <Slider
                value={hyperparams.max_depth}
                min={3}
                max={15}
                step={1}
                onChange={(e, value) => handleHyperparamChange('max_depth', value)}
                valueLabelDisplay="auto"
              />
              <Typography variant="body2" color="text.secondary">
                Current: {hyperparams.max_depth}
              </Typography>
            </Grid>

            <Grid item xs={12} md={6}>
              <Typography gutterBottom>Batch Size (Neural Network)</Typography>
              <Slider
                value={hyperparams.batch_size}
                min={16}
                max={128}
                step={16}
                onChange={(e, value) => handleHyperparamChange('batch_size', value)}
                valueLabelDisplay="auto"
              />
              <Typography variant="body2" color="text.secondary">
                Current: {hyperparams.batch_size}
              </Typography>
            </Grid>
          </Grid>

          <Box sx={{ mt: 3, textAlign: 'center', display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Button
              variant="outlined"
              color="info"
              onClick={handleUpdateNASAData}
              startIcon={<TrendingUp />}
              size="large"
            >
              Update NASA Dataset
            </Button>
            <Button
              variant="contained"
              color="secondary"
              onClick={handleRetrainModel}
              disabled={isTraining}
              startIcon={<Psychology />}
              size="large"
            >
              {isTraining ? 'Training Model...' : 'Retrain with New Parameters'}
            </Button>
          </Box>
        </CardContent>
      </Card>

      {/* Training History */}
      {trainingHistory && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Training History
            </Typography>
            <Plot
              data={[
                {
                  x: trainingHistory.epochs,
                  y: trainingHistory.accuracy,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Accuracy',
                  line: { color: '#1976d2' }
                },
                {
                  x: trainingHistory.epochs,
                  y: trainingHistory.loss,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Loss',
                  yaxis: 'y2',
                  line: { color: '#d32f2f' }
                }
              ]}
              layout={{
                title: 'Training Progress',
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Accuracy', side: 'left' },
                yaxis2: { title: 'Loss', side: 'right', overlaying: 'y' },
                showlegend: true,
                height: 400
              }}
              style={{ width: '100%' }}
            />
          </CardContent>
        </Card>
      )}
    </Box>
  );

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          <Psychology sx={{ mr: 2, verticalAlign: 'middle' }} />
          AI Model Management
        </Typography>
        
        <FormControlLabel
          control={
            <Switch
              checked={userMode === 'researcher'}
              onChange={(e) => setUserMode(e.target.checked ? 'researcher' : 'novice')}
            />
          }
          label={userMode === 'researcher' ? 'Researcher Mode' : 'Novice Mode'}
        />
      </Box>

      {isTraining && (
        <Box sx={{ mb: 3 }}>
          <LinearProgress />
          <Typography variant="body2" align="center" sx={{ mt: 1 }}>
            Training AI model with new parameters... This may take a few minutes.
          </Typography>
        </Box>
      )}

      {userMode === 'novice' ? renderNoviceInterface() : renderResearcherInterface()}
    </Container>
  );
};

export default ModelManagement;