import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  Card,
  CardContent
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';

const Upload = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState('');

  const handleFileSelect = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadStatus('Please select a file first');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setUploadStatus('Uploading...');
      // Upload logic here
      setUploadStatus('Upload successful!');
    } catch (error) {
      setUploadStatus('Upload failed: ' + error.message);
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Data Upload
      </Typography>
      
      <Paper sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          Upload Exoplanet Data Files
        </Typography>
        
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="body1" gutterBottom>
              Supported file formats: CSV, JSON, FITS
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Upload your light curve data, stellar parameters, or observational data for analysis.
            </Typography>
          </CardContent>
        </Card>

        <Box sx={{ textAlign: 'center', mb: 3 }}>
          <input
            accept=".csv,.json,.fits"
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
            >
              Select File
            </Button>
          </label>
          
          {selectedFile && (
            <Typography variant="body2">
              Selected: {selectedFile.name}
            </Typography>
          )}
        </Box>

        <Button
          variant="contained"
          onClick={handleUpload}
          disabled={!selectedFile}
          fullWidth
        >
          Upload and Process
        </Button>

        {uploadStatus && (
          <Alert 
            severity={uploadStatus.includes('successful') ? 'success' : 'info'} 
            sx={{ mt: 2 }}
          >
            {uploadStatus}
          </Alert>
        )}
      </Paper>
    </Container>
  );
};

export default Upload;