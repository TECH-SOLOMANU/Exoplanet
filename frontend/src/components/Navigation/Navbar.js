import React, { useState } from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material';
import {
  Home,
  TravelExplore,
  CloudUpload,
  Psychology,
  ShowChart,
  Info,
  Menu as MenuIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';

const Navbar = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [anchorEl, setAnchorEl] = useState(null);

  const menuItems = [
    { label: 'Dashboard', path: '/', icon: <Home /> },
    { label: 'Predictions', path: '/predictions', icon: <TravelExplore /> },
    { label: 'Upload Data', path: '/upload', icon: <CloudUpload /> },
    { label: 'Models', path: '/models', icon: <Psychology /> },
    { label: 'Light Curves', path: '/light-curves', icon: <ShowChart /> },
    { label: 'About', path: '/about', icon: <Info /> },
  ];

  const handleMenuClick = (path) => {
    navigate(path);
    setAnchorEl(null);
  };

  const handleMenuOpen = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleMenuClose = () => {
    setAnchorEl(null);
  };

  return (
    <AppBar 
      position="sticky" 
      sx={{ 
        background: 'linear-gradient(90deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
      }}
    >
      <Toolbar>
        {/* Logo and Title */}
        <Box sx={{ display: 'flex', alignItems: 'center', flexGrow: 1 }}>
          <TravelExplore sx={{ mr: 2, fontSize: 32, color: '#42a5f5' }} />
          <Box>
            <Typography variant="h6" component="div" sx={{ fontWeight: 600 }}>
              Exoplanet Detection Platform
            </Typography>
            <Typography variant="caption" sx={{ color: '#b0b0b0', fontSize: '0.7rem' }}>
              NASA Space Apps Challenge 2025
            </Typography>
          </Box>
        </Box>

        {/* Desktop Navigation */}
        <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 1 }}>
          {menuItems.map((item) => (
            <Button
              key={item.path}
              startIcon={item.icon}
              onClick={() => navigate(item.path)}
              sx={{
                color: location.pathname === item.path ? '#42a5f5' : 'white',
                fontWeight: location.pathname === item.path ? 600 : 400,
                borderBottom: location.pathname === item.path ? '2px solid #42a5f5' : 'none',
                borderRadius: 0,
                '&:hover': {
                  background: 'rgba(66, 165, 245, 0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>

        {/* Mobile Navigation */}
        <Box sx={{ display: { xs: 'flex', md: 'none' } }}>
          <IconButton
            size="large"
            edge="start"
            color="inherit"
            onClick={handleMenuOpen}
          >
            <MenuIcon />
          </IconButton>
          <Menu
            anchorEl={anchorEl}
            open={Boolean(anchorEl)}
            onClose={handleMenuClose}
            sx={{
              '& .MuiPaper-root': {
                background: '#1a1a2e',
                border: '1px solid #2c2c54',
              },
            }}
          >
            {menuItems.map((item) => (
              <MenuItem
                key={item.path}
                onClick={() => handleMenuClick(item.path)}
                sx={{
                  color: location.pathname === item.path ? '#42a5f5' : 'white',
                  '&:hover': {
                    background: 'rgba(66, 165, 245, 0.1)',
                  },
                }}
              >
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  {item.icon}
                  {item.label}
                </Box>
              </MenuItem>
            ))}
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;