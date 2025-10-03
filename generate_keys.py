#!/usr/bin/env python3
"""
Secret Key Generator for NASA Exoplanet Detection Platform
Generates cryptographically secure keys for production deployment
"""

import secrets
import string
import base64
import os
from datetime import datetime

def generate_secret_key(length=32):
    """Generate a cryptographically secure secret key"""
    return secrets.token_urlsafe(length)

def generate_jwt_secret(length=64):
    """Generate a JWT secret key"""
    return secrets.token_urlsafe(length)

def generate_hex_key(length=32):
    """Generate a hex-encoded secret key"""
    return secrets.token_hex(length)

def generate_base64_key(length=32):
    """Generate a base64-encoded secret key"""
    key_bytes = secrets.token_bytes(length)
    return base64.b64encode(key_bytes).decode('utf-8')

def generate_custom_key(length=32):
    """Generate a custom alphanumeric key"""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def main():
    print("=" * 60)
    print("🚀 NASA Exoplanet Detection Platform")
    print("🔐 Secret Key Generator")
    print("=" * 60)
    print()
    
    print("📋 Generated Secret Keys:")
    print("-" * 40)
    
    # Generate different types of keys
    secret_key = generate_secret_key(32)
    jwt_secret = generate_jwt_secret(64)
    hex_key = generate_hex_key(32)
    base64_key = generate_base64_key(32)
    custom_key = generate_custom_key(48)
    
    print(f"🔑 SECRET_KEY (URL-safe, 32 bytes):")
    print(f"   {secret_key}")
    print()
    
    print(f"🎫 JWT_SECRET_KEY (URL-safe, 64 bytes):")
    print(f"   {jwt_secret}")
    print()
    
    print(f"🔐 HEX_KEY (Hexadecimal, 32 bytes):")
    print(f"   {hex_key}")
    print()
    
    print(f"📝 BASE64_KEY (Base64 encoded, 32 bytes):")
    print(f"   {base64_key}")
    print()
    
    print(f"🛡️ CUSTOM_KEY (Mixed chars, 48 bytes):")
    print(f"   {custom_key}")
    print()
    
    # Generate .env content
    env_content = f"""# NASA Exoplanet Detection Platform - Secret Keys
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Application Secret Key (32 bytes, URL-safe)
SECRET_KEY={secret_key}

# JWT Secret Key (64 bytes, URL-safe)
JWT_SECRET_KEY={jwt_secret}

# Additional Security Keys
ENCRYPTION_KEY={hex_key}
SESSION_SECRET={base64_key}

# NASA API Configuration
NASA_API_KEY=your_nasa_api_key_here

# Database Configuration
MONGODB_URL=mongodb://localhost:27017/exoplanet_db
DATABASE_NAME=exoplanet_production

# Gemini AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Application Settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# CORS Settings
CORS_ORIGINS=http://localhost:3000,https://your-domain.com
"""
    
    # Write to .env file
    with open('.env.generated', 'w') as f:
        f.write(env_content)
    
    print("📄 Complete .env file generated as '.env.generated'")
    print("🔄 Copy the contents to your .env file")
    print()
    
    print("⚠️  SECURITY NOTES:")
    print("- Never commit .env files to version control")
    print("- Use different keys for development/production")
    print("- Rotate keys regularly in production")
    print("- Store keys securely (AWS Secrets Manager, etc.)")
    print()
    
    print("🚀 For NASA Space Apps Challenge:")
    print("- Use the SECRET_KEY and JWT_SECRET_KEY above")
    print("- Get Gemini API key from: https://aistudio.google.com/")
    print("- Get NASA API key from: https://api.nasa.gov/")
    print()

if __name__ == "__main__":
    main()