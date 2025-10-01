import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  Card,
  CardContent,
  LinearProgress,
  Grid,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon
} from '@mui/material';
import { 
  CloudUpload, 
  Assessment, 
  Science, 
  CheckCircle, 
  Schedule,
  DataUsage,
  TrendingUp
} from '@mui/icons-material';
import Plot from 'react-plotly.js';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [lightCurveId, setLightCurveId] = useState(null);
  const [advancedAnalysis, setAdvancedAnalysis] = useState(null);
  const [isRunningAdvanced, setIsRunningAdvanced] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setUploadStatus('');
    setAnalysisResult(null);
    setLightCurveId(null);
    setAdvancedAnalysis(null);
    setIsRunningAdvanced(false);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setIsUploading(true);
      setUploadStatus('Uploading and processing...');
      
      const response = await fetch('http://localhost:800/api/v1/upload/light-curve', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setUploadStatus('Upload successful! Analysis in progress...');
        setLightCurveId(result.light_curve_id);
        
        // Set initial analysis result with upload data
        setAnalysisResult({
          ...result,
          analysis_status: 'processing'
        });
        
        // Start polling for analysis results
        setTimeout(() => pollAnalysisResults(result.light_curve_id), 2000);
      } else {
        const errorData = await response.json();
        setUploadStatus(`Upload failed: ${errorData.detail || 'Unknown error'}`);
      }
    } catch (error) {
      setUploadStatus('Upload failed: ' + error.message);
    } finally {
      setIsUploading(false);
    }
  };

  const pollAnalysisResults = async (lcId) => {
    try {
      const response = await fetch(`http://localhost:800/api/v1/upload/light-curve/${lcId}/analysis`);
      if (response.ok) {
        const result = await response.json();
        console.log('Analysis result:', result); // Debug logging
        
        // Only update if we have valid data
        if (result && typeof result === 'object') {
          setAnalysisResult(prevResult => {
            // Merge with previous result, keeping existing data
            const merged = {
              ...prevResult,
              ...result,
              // Preserve some fields that shouldn't disappear
              data_points: result.data_points || prevResult?.data_points,
              time_span: result.time_span || prevResult?.time_span,
              filename: result.filename || prevResult?.filename
            };
            return merged;
          });
          
          // Check status and continue polling if needed
          if (result.analysis_status === 'pending' || result.analysis_status === 'processing') {
            // Continue polling but with longer intervals to reduce load
            setTimeout(() => pollAnalysisResults(lcId), 5000);
          } else if (result.analysis_status === 'completed') {
            setUploadStatus('Analysis completed successfully!');
          } else if (result.analysis_status === 'failed') {
            setUploadStatus('Analysis failed. Please try reanalyzing.');
          }
        }
      } else {
        console.error('Failed to fetch analysis:', response.status);
        // Only retry on 404 initially, but stop retrying after 30 seconds
        if (response.status === 404) {
          setTimeout(() => pollAnalysisResults(lcId), 3000);
        }
      }
    } catch (error) {
      console.error('Failed to fetch analysis results:', error);
      // Retry on network errors but with exponential backoff
      setTimeout(() => pollAnalysisResults(lcId), 5000);
    }
  };

  const triggerAdvancedAnalysis = async () => {
    if (!lightCurveId) return;
    
    try {
      setIsRunningAdvanced(true);
      setUploadStatus('Starting advanced analysis...');
      
      const response = await fetch(`http://localhost:800/api/v1/upload/light-curve/${lightCurveId}/advanced-analysis`, {
        method: 'POST',
      });
      
      if (response.ok) {
        const result = await response.json();
        setUploadStatus('Advanced analysis initiated. This may take 2-3 minutes...');
        
        // Poll for advanced analysis results
        setTimeout(() => pollAdvancedAnalysisResults(lightCurveId), 5000);
      } else {
        setUploadStatus('Failed to start advanced analysis');
      }
    } catch (error) {
      setUploadStatus('Advanced analysis failed: ' + error.message);
    } finally {
      setIsRunningAdvanced(false);
    }
  };

  const pollAdvancedAnalysisResults = async (lcId) => {
    try {
      const response = await fetch(`http://localhost:800/api/v1/upload/light-curve/${lcId}/advanced-analysis`);
      if (response.ok) {
        const result = await response.json();
        
        if (result.status === 'processing') {
          // Continue polling
          setTimeout(() => pollAdvancedAnalysisResults(lcId), 10000);
        } else if (result.status === 'completed') {
          setAdvancedAnalysis(result);
          setUploadStatus('Advanced analysis completed!');
        } else if (result.status === 'failed') {
          setUploadStatus(`Advanced analysis failed: ${result.error}`);
        }
      }
    } catch (error) {
      console.error('Failed to fetch advanced analysis results:', error);
    }
  };

  const triggerReanalysis = async () => {
    if (!lightCurveId) return;
    
    try {
      setUploadStatus('Reanalysis initiated...');
      const response = await fetch(`http://localhost:800/api/v1/upload/light-curve/${lightCurveId}/reanalyze`, {
        method: 'POST',
      });
      
      if (response.ok) {
        setTimeout(() => pollAnalysisResults(lightCurveId), 3000);
      }
    } catch (error) {
      setUploadStatus('Reanalysis failed: ' + error.message);
    }
  };

  const renderLightCurvePlot = () => {
    if (!analysisResult?.sample_data) return null;

    const { time, flux } = analysisResult.sample_data;
    
    const plotData = [{
      x: time,
      y: flux,
      type: 'scatter',
      mode: 'lines+markers',
      marker: { 
        size: 4, 
        color: '#1976d2',
        line: { width: 0.5, color: '#ffffff' }
      },
      line: { color: '#1976d2', width: 1 },
      name: 'Light Curve'
    }];

    // Add transit events if available
    if (analysisResult.transit_analysis?.transit_events) {
      analysisResult.transit_analysis.transit_events.forEach((event, index) => {
        plotData.push({
          x: [event.start_time, event.end_time, event.end_time, event.start_time, event.start_time],
          y: [0.98, 0.98, 1.02, 1.02, 0.98],
          fill: 'toself',
          fillcolor: 'rgba(255, 0, 0, 0.1)',
          line: { color: 'red', width: 1 },
          name: `Transit ${index + 1}`,
          hovertemplate: `Transit Event<br>Depth: ${(event.depth * 100).toFixed(2)}%<br>Duration: ${(event.duration * 24).toFixed(1)} hours<extra></extra>`
        });
      });
    }

    const layout = {
      title: 'Light Curve Analysis',
      xaxis: { 
        title: 'Time (days)',
        showgrid: true,
        gridcolor: '#f0f0f0'
      },
      yaxis: { 
        title: 'Relative Flux',
        showgrid: true,
        gridcolor: '#f0f0f0'
      },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: '#ffffff',
      font: { family: 'Roboto, sans-serif' },
      margin: { t: 50, r: 30, b: 50, l: 60 },
      showlegend: true,
      legend: { x: 1, y: 1 }
    };

    const config = {
      displayModeBar: true,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
      displaylogo: false,
      responsive: true
    };

    return (
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Interactive Light Curve Visualization
          </Typography>
          <Plot
            data={plotData}
            layout={layout}
            config={config}
            style={{ width: '100%', height: '400px' }}
          />
        </CardContent>
      </Card>
    );
  };

  const renderAnalysisResults = () => {
    if (!analysisResult) return null;

    const transitAnalysis = analysisResult.transit_analysis;
    const isProcessing = analysisResult.analysis_status === 'pending' || analysisResult.analysis_status === 'processing';
    
    return (
      <Box sx={{ mt: 3 }}>
        {/* Show processing indicator */}
        {isProcessing && (
          <Card sx={{ mb: 2, bgcolor: 'info.light' }}>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <LinearProgress sx={{ flexGrow: 1, mr: 2 }} />
                <Typography variant="body2">
                  {analysisResult.analysis_status === 'pending' ? 'Queued for analysis...' : 'Analyzing light curve...'}
                </Typography>
              </Box>
            </CardContent>
          </Card>
        )}
        
        {/* Basic Statistics */}
        <Card sx={{ mb: 2 }}>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              <DataUsage sx={{ mr: 1, verticalAlign: 'middle' }} />
              Data Summary
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} md={3}>
                <Typography variant="body2" color="text.secondary">Data Points</Typography>
                <Typography variant="h6">{analysisResult.data_points?.toLocaleString() || 'Processing...'}</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="body2" color="text.secondary">Time Span</Typography>
                <Typography variant="h6">{analysisResult.time_span?.toFixed(1) || 'Processing...'} days</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="body2" color="text.secondary">File Size</Typography>
                <Typography variant="h6">{(selectedFile?.size / 1024)?.toFixed(1) || '0'} KB</Typography>
              </Grid>
              <Grid item xs={6} md={3}>
                <Typography variant="body2" color="text.secondary">Status</Typography>
                <Chip 
                  label={analysisResult.analysis_status || 'Processing'} 
                  color={
                    analysisResult.analysis_status === 'completed' ? 'success' : 
                    analysisResult.analysis_status === 'failed' ? 'error' : 'warning'
                  }
                  size="small"
                />
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Transit Detection Results */}
        {transitAnalysis && (
          <Card sx={{ mb: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                <Science sx={{ mr: 1, verticalAlign: 'middle' }} />
                AI Transit Detection Results
              </Typography>
              
              <Box sx={{ mb: 2 }}>
                <Grid container spacing={3}>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color={transitAnalysis.planet_detected ? 'success.main' : 'text.secondary'}>
                        {transitAnalysis.planet_detected ? '✓' : '✗'}
                      </Typography>
                      <Typography variant="body2">
                        Planet Detection
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="primary">
                        {(transitAnalysis.confidence * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2">
                        Confidence Score
                      </Typography>
                    </Box>
                  </Grid>
                  <Grid item xs={12} md={4}>
                    <Box sx={{ textAlign: 'center' }}>
                      <Typography variant="h4" color="info.main">
                        {transitAnalysis.transit_events?.length || 0}
                      </Typography>
                      <Typography variant="body2">
                        Transit Events
                      </Typography>
                    </Box>
                  </Grid>
                </Grid>
              </Box>

              <Divider sx={{ my: 2 }} />

              {/* Planet Parameters */}
              {transitAnalysis.planet_parameters && (
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle1" gutterBottom>
                    Estimated Planet Parameters
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">Radius Ratio</Typography>
                      <Typography variant="body1">
                        {transitAnalysis.planet_parameters.estimated_radius_ratio?.toFixed(4) || 'N/A'}
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" color="text.secondary">Transit Duration</Typography>
                      <Typography variant="body1">
                        {transitAnalysis.planet_parameters.transit_duration_hours?.toFixed(1) || 'N/A'} hours
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              )}

              {/* Recommendations */}
              {transitAnalysis.recommendations && (
                <Box>
                  <Typography variant="subtitle1" gutterBottom>
                    Analysis Recommendations
                  </Typography>
                  <List dense>
                    {transitAnalysis.recommendations.map((rec, index) => (
                      <ListItem key={index}>
                        <ListItemIcon>
                          <TrendingUp fontSize="small" />
                        </ListItemIcon>
                        <ListItemText primary={rec} />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </CardContent>
          </Card>
        )}

        {/* Reanalysis and Advanced Analysis Options */}
        {analysisResult.analysis_status === 'completed' && (
          <Box sx={{ textAlign: 'center', mt: 2 }}>
            <Button 
              variant="outlined" 
              onClick={triggerReanalysis}
              startIcon={<Assessment />}
              sx={{ mr: 2 }}
            >
              Run Standard Reanalysis
            </Button>
            <Button 
              variant="contained" 
              onClick={triggerAdvancedAnalysis}
              disabled={isRunningAdvanced}
              startIcon={<Science />}
              color="secondary"
            >
              {isRunningAdvanced ? 'Starting...' : 'Run Advanced Analysis'}
            </Button>
          </Box>
        )}
      </Box>
    );
  };

  const renderAdvancedAnalysisResults = () => {
    if (!advancedAnalysis || advancedAnalysis.status !== 'completed') return null;

    return (
      <Box sx={{ mt: 3 }}>
        <Card>
          <CardContent>
            <Typography variant="h5" gutterBottom>
              <Science sx={{ mr: 1, verticalAlign: 'middle' }} />
              Advanced Analysis Results
            </Typography>
            
            {/* Overall Assessment */}
            {advancedAnalysis.overall_assessment && (
              <Card sx={{ mb: 3, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Overall Assessment
                  </Typography>
                  <Typography variant="h4" sx={{ mb: 2 }}>
                    {(advancedAnalysis.overall_assessment.combined_confidence * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body1" sx={{ mb: 2 }}>
                    <strong>Recommendation:</strong> {advancedAnalysis.overall_assessment.recommendation}
                  </Typography>
                  <Typography variant="body2">
                    <strong>Data Quality:</strong> {advancedAnalysis.overall_assessment.data_quality}
                  </Typography>
                  
                  {advancedAnalysis.overall_assessment.key_findings && (
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>Key Findings:</Typography>
                      <List dense>
                        {advancedAnalysis.overall_assessment.key_findings.map((finding, index) => (
                          <ListItem key={index}>
                            <ListItemIcon>
                              <CheckCircle fontSize="small" />
                            </ListItemIcon>
                            <ListItemText primary={finding} />
                          </ListItem>
                        ))}
                      </List>
                    </Box>
                  )}
                </CardContent>
              </Card>
            )}

            <Grid container spacing={3}>
              {/* Machine Learning Classification */}
              {advancedAnalysis.ml_classification && (
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        🤖 Machine Learning Classification
                      </Typography>
                      <Box sx={{ textAlign: 'center', mb: 2 }}>
                        <Typography variant="h3" color="primary">
                          {(advancedAnalysis.ml_classification.planet_probability * 100).toFixed(1)}%
                        </Typography>
                        <Typography variant="body2">Planet Probability</Typography>
                      </Box>
                      <Typography variant="body1" sx={{ mb: 1 }}>
                        <strong>Classification:</strong> {advancedAnalysis.ml_classification.classification}
                      </Typography>
                      <Typography variant="body2">
                        <strong>Model:</strong> {advancedAnalysis.ml_classification.model_version}
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Periodicity Analysis */}
              {advancedAnalysis.frequency_analysis && (
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        📊 Periodicity Analysis
                      </Typography>
                      <Typography variant="body1" sx={{ mb: 2 }}>
                        <strong>Periodicity Detected:</strong> {
                          advancedAnalysis.frequency_analysis.periodicity_detected ? 'Yes' : 'No'
                        }
                      </Typography>
                      
                      {advancedAnalysis.frequency_analysis.dominant_periods && 
                       advancedAnalysis.frequency_analysis.dominant_periods.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" gutterBottom>
                            Dominant Periods:
                          </Typography>
                          {advancedAnalysis.frequency_analysis.dominant_periods.slice(0, 3).map((period, index) => (
                            <Typography key={index} variant="body2">
                              • {period.period_days.toFixed(2)} days (Power: {period.power.toFixed(3)})
                            </Typography>
                          ))}
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Statistical Analysis */}
              {advancedAnalysis.statistical_analysis && (
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        📈 Statistical Analysis
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">RMS Variability</Typography>
                          <Typography variant="body1">
                            {(advancedAnalysis.statistical_analysis.std * 100).toFixed(4)}%
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Skewness</Typography>
                          <Typography variant="body1">
                            {advancedAnalysis.statistical_analysis.skewness?.toFixed(3) || 'N/A'}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Kurtosis</Typography>
                          <Typography variant="body1">
                            {advancedAnalysis.statistical_analysis.kurtosis?.toFixed(3) || 'N/A'}
                          </Typography>
                        </Grid>
                        <Grid item xs={6}>
                          <Typography variant="body2" color="text.secondary">Linear Trend</Typography>
                          <Typography variant="body1">
                            {advancedAnalysis.statistical_analysis.linear_trend?.toFixed(6) || 'N/A'}
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              )}

              {/* Data Quality */}
              {advancedAnalysis.quality_metrics && (
                <Grid item xs={12} md={6}>
                  <Card sx={{ height: '100%' }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        ✅ Data Quality Assessment
                      </Typography>
                      <Box sx={{ textAlign: 'center', mb: 2 }}>
                        <Typography variant="h3" color="success.main">
                          {(advancedAnalysis.quality_metrics.overall_quality_score * 100).toFixed(0)}%
                        </Typography>
                        <Typography variant="body2">Overall Quality Score</Typography>
                      </Box>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Completeness:</strong> {(advancedAnalysis.quality_metrics.completeness * 100).toFixed(1)}%
                      </Typography>
                      <Typography variant="body2" sx={{ mb: 1 }}>
                        <strong>Outliers:</strong> {(advancedAnalysis.quality_metrics.outlier_fraction * 100).toFixed(2)}%
                      </Typography>
                      <Typography variant="body2">
                        <strong>Noise Level:</strong> {(advancedAnalysis.quality_metrics.noise_level * 100).toFixed(3)}%
                      </Typography>
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>

            {/* Frequency Spectrum Visualization */}
            {advancedAnalysis.frequency_analysis?.frequency_grid && 
             advancedAnalysis.frequency_analysis?.power_spectrum && (
              <Card sx={{ mt: 3 }}>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Frequency Power Spectrum
                  </Typography>
                  <Plot
                    data={[{
                      x: advancedAnalysis.frequency_analysis.frequency_grid,
                      y: advancedAnalysis.frequency_analysis.power_spectrum,
                      type: 'scatter',
                      mode: 'lines',
                      line: { color: '#ff6b35', width: 1 },
                      name: 'Power Spectrum'
                    }]}
                    layout={{
                      title: 'Lomb-Scargle Periodogram',
                      xaxis: { 
                        title: 'Frequency (cycles/day)',
                        showgrid: true,
                        gridcolor: '#f0f0f0'
                      },
                      yaxis: { 
                        title: 'Normalized Power',
                        showgrid: true,
                        gridcolor: '#f0f0f0'
                      },
                      plot_bgcolor: '#fafafa',
                      paper_bgcolor: '#ffffff',
                      font: { family: 'Roboto, sans-serif' },
                      margin: { t: 50, r: 30, b: 50, l: 60 }
                    }}
                    config={{
                      displayModeBar: true,
                      displaylogo: false,
                      responsive: true
                    }}
                    style={{ width: '100%', height: '300px' }}
                  />
                </CardContent>
              </Card>
            )}
          </CardContent>
        </Card>
      </Box>
    );
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        <CloudUpload sx={{ mr: 2, verticalAlign: 'middle' }} />
        Light Curve Analysis Platform
      </Typography>
      <Typography variant="subtitle1" color="text.secondary" gutterBottom>
        Upload your light curve data for AI-powered exoplanet detection
      </Typography>
      
      <Paper sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Upload Light Curve Data
        </Typography>
        
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="body1" gutterBottom>
              <strong>Supported Formats:</strong> CSV, JSON, TXT
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Upload time-series photometry data from telescopes like Kepler, TESS, or ground-based observations.
              Expected columns: time (days/JD) and flux/brightness measurements.
            </Typography>
          </CardContent>
        </Card>

        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <input
            accept=".csv,.json,.txt,.dat"
            style={{ display: 'none' }}
            id="file-upload"
            type="file"
            onChange={handleFileSelect}
          />
          <label htmlFor="file-upload">
            <Button
              variant="outlined"
              component="span"
              startIcon={<CloudUpload />}
              sx={{ mb: 2 }}
              size="large"
            >
              Select Light Curve File
            </Button>
          </label>
          
          {selectedFile && (
            <Box sx={{ mt: 2 }}>
              <Chip 
                label={selectedFile.name} 
                color="primary" 
                onDelete={() => setSelectedFile(null)}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Size: {(selectedFile.size / 1024).toFixed(1)} KB
              </Typography>
            </Box>
          )}
        </Box>

        <Button
          variant="contained"
          onClick={handleUpload}
          disabled={!selectedFile || isUploading}
          fullWidth
          size="large"
          startIcon={isUploading ? <Schedule /> : <Assessment />}
        >
          {isUploading ? 'Processing...' : 'Upload & Analyze'}
        </Button>

        {isUploading && (
          <Box sx={{ mt: 2 }}>
            <LinearProgress />
            <Typography variant="body2" align="center" sx={{ mt: 1 }}>
              Processing your light curve data...
            </Typography>
          </Box>
        )}

        {uploadStatus && (
          <Alert 
            severity={
              uploadStatus.includes('successful') || uploadStatus.includes('completed') 
                ? 'success' 
                : uploadStatus.includes('failed') 
                ? 'error' 
                : 'info'
            } 
            sx={{ mt: 2 }}
          >
            {uploadStatus}
          </Alert>
        )}
      </Paper>

      {/* Visualization and Results */}
      {renderLightCurvePlot()}
      {renderAnalysisResults()}
      {renderAdvancedAnalysisResults()}
    </Container>
  );
};

export default Upload;