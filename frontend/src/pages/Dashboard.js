import React, { useState, useEffect } from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import {
  TravelExplore,
  Science,
  Psychology,
  CloudDownload,
  Refresh,
  Timeline,
} from '@mui/icons-material';
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend } from 'recharts';
import axios from 'axios';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [latestDiscoveries, setLatestDiscoveries] = useState([]);
  const [fetchStatus, setFetchStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      
      // Load all dashboard data in parallel
      const [statsResponse, discoveriesResponse, statusResponse] = await Promise.all([
        axios.get('http://localhost:800/api/v1/nasa/stats'),
        axios.get('http://localhost:800/api/v1/nasa/latest'),
        axios.get('http://localhost:800/api/v1/nasa/status'),
      ]);

      setStats(statsResponse.data);
      setLatestDiscoveries(discoveriesResponse.data.latest_discoveries || []);
      setFetchStatus(statusResponse.data);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFetchNASAData = async () => {
    try {
      setFetching(true);
      await axios.get('http://localhost:800/api/v1/nasa/fetch');
      
      // Reload data after a short delay
      setTimeout(() => {
        loadDashboardData();
        setFetching(false);
      }, 2000);
    } catch (error) {
      console.error('Failed to fetch NASA data:', error);
      setFetching(false);
    }
  };

  const formatStatusData = (statusDistribution) => {
    if (!statusDistribution) return [];
    
    return Object.entries(statusDistribution).map(([status, count]) => ({
      name: status.replace('_', ' '),
      value: count,
      color: status === 'CONFIRMED' ? '#4caf50' : 
             status === 'CANDIDATE' ? '#ff9800' : '#f44336'
    }));
  };

  const formatYearData = (yearDistribution) => {
    if (!yearDistribution) return [];
    
    return yearDistribution.map(item => ({
      year: item.year,
      discoveries: item.discoveries
    }));
  };

  if (loading) {
    return (
      <Container maxWidth="lg" sx={{ mt: 4, display: 'flex', justifyContent: 'center' }}>
        <CircularProgress size={60} />
      </Container>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      {/* Header */}
      <Box sx={{ mb: 4, textAlign: 'center' }}>
        <Typography variant="h3" component="h1" gutterBottom sx={{ fontWeight: 600 }}>
          🌟 Exoplanet Detection Dashboard
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
          NASA Space Apps Challenge 2025 - Real-time Exoplanet Discovery Platform
        </Typography>
        
        <Button
          variant="contained"
          startIcon={fetching ? <CircularProgress size={20} /> : <CloudDownload />}
          onClick={handleFetchNASAData}
          disabled={fetching}
          sx={{ mr: 2 }}
        >
          {fetching ? 'Fetching...' : 'Fetch Latest NASA Data'}
        </Button>
        
        <Button
          variant="outlined"
          startIcon={<Refresh />}
          onClick={loadDashboardData}
        >
          Refresh Dashboard
        </Button>
      </Box>

      {/* Fetch Status Alert */}
      {fetchStatus && (
        <Alert 
          severity={fetchStatus.status === 'completed' ? 'success' : 
                   fetchStatus.status === 'failed' ? 'error' : 'info'}
          sx={{ mb: 3 }}
        >
          <Typography variant="body2">
            <strong>Last NASA Data Fetch:</strong> {fetchStatus.message}
            {fetchStatus.records_fetched > 0 && (
              <span> ({fetchStatus.records_fetched} records)</span>
            )}
          </Typography>
        </Alert>
      )}

      {/* Statistics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <TravelExplore sx={{ fontSize: 40, color: '#4caf50', mb: 1 }} />
              <Typography variant="h4" component="div" sx={{ fontWeight: 600 }}>
                {stats?.total_exoplanets || 0}
              </Typography>
              <Typography color="text.secondary">
                Total Exoplanets
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Science sx={{ fontSize: 40, color: '#2196f3', mb: 1 }} />
              <Typography variant="h4" component="div" sx={{ fontWeight: 600 }}>
                {stats?.status_distribution?.CONFIRMED || 0}
              </Typography>
              <Typography color="text.secondary">
                Confirmed Planets
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Psychology sx={{ fontSize: 40, color: '#ff9800', mb: 1 }} />
              <Typography variant="h4" component="div" sx={{ fontWeight: 600 }}>
                {stats?.status_distribution?.CANDIDATE || 0}
              </Typography>
              <Typography color="text.secondary">
                Planet Candidates
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Timeline sx={{ fontSize: 40, color: '#9c27b0', mb: 1 }} />
              <Typography variant="h4" component="div" sx={{ fontWeight: 600 }}>
                {stats?.database_collections?.predictions || 0}
              </Typography>
              <Typography color="text.secondary">
                AI Predictions
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Status Distribution Pie Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Exoplanet Status Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={formatStatusData(stats?.status_distribution)}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {formatStatusData(stats?.status_distribution).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Discovery Timeline Bar Chart */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Discovery Timeline
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={formatYearData(stats?.recent_years_distribution)}>
                    <XAxis dataKey="year" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="discoveries" fill="#42a5f5" />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Latest Discoveries */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🚀 Latest Exoplanet Discoveries
          </Typography>
          {latestDiscoveries.length === 0 ? (
            <Typography color="text.secondary">
              No recent discoveries available. Try fetching the latest NASA data.
            </Typography>
          ) : (
            <Grid container spacing={2}>
              {latestDiscoveries.map((planet, index) => (
                <Grid item xs={12} sm={6} md={4} key={index}>
                  <Card variant="outlined">
                    <CardContent>
                      <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 1 }}>
                        {planet.pl_name}
                      </Typography>
                      
                      <Box sx={{ mb: 2 }}>
                        <Chip 
                          label={planet.pl_status} 
                          color="success" 
                          size="small" 
                          sx={{ mb: 1 }}
                        />
                        {planet.pl_disc && (
                          <Chip 
                            label={`Discovered: ${planet.pl_disc}`} 
                            variant="outlined" 
                            size="small" 
                            sx={{ mb: 1, ml: 1 }}
                          />
                        )}
                      </Box>

                      <Box sx={{ fontSize: '0.875rem', color: 'text.secondary' }}>
                        {planet.pl_orbper && (
                          <Typography variant="body2">
                            Period: {planet.pl_orbper.toFixed(2)} days
                          </Typography>
                        )}
                        {planet.pl_rade && (
                          <Typography variant="body2">
                            Radius: {planet.pl_rade.toFixed(2)} Earth radii
                          </Typography>
                        )}
                        {planet.pl_facility && (
                          <Typography variant="body2">
                            Facility: {planet.pl_facility}
                          </Typography>
                        )}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </CardContent>
      </Card>
    </Container>
  );
};

export default Dashboard;