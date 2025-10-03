# Alternative Methods to Generate Secret Keys

## Method 2: PowerShell (Windows)
```powershell
# Generate SECRET_KEY (32 bytes)
[System.Web.Security.Membership]::GeneratePassword(43, 0)

# Or using .NET crypto
$bytes = New-Object byte[] 32
$rng = [System.Security.Cryptography.RNGCryptoServiceProvider]::Create()
$rng.GetBytes($bytes)
[Convert]::ToBase64String($bytes)
```

## Method 3: OpenSSL (Cross-platform)
```bash
# Generate SECRET_KEY (32 bytes, base64)
openssl rand -base64 32

# Generate JWT_SECRET_KEY (64 bytes, base64)
openssl rand -base64 64

# Generate hex key
openssl rand -hex 32
```

## Method 4: Online Generators (Use with caution)
- https://generate-secret.now.sh/32
- https://www.allkeysgenerator.com/Random/Security-Encryption-Key-Generator.aspx

⚠️ **Note**: For production, always generate keys locally for security!

## Method 5: Node.js/JavaScript
```javascript
// In Node.js
const crypto = require('crypto');
console.log(crypto.randomBytes(32).toString('base64'));
```

## Method 6: Django-style Secret Key
```python
from django.core.management.utils import get_random_secret_key
print(get_random_secret_key())
```

## Method 7: UUID-based (Simple but less secure)
```python
import uuid
secret = str(uuid.uuid4()).replace('-', '') + str(uuid.uuid4()).replace('-', '')
print(secret)
```

## NASA Space Apps Challenge Setup
1. Use the generated keys from the Python script above
2. Create your .env file with these keys
3. Get API keys from:
   - Gemini: https://aistudio.google.com/
   - NASA: https://api.nasa.gov/
   - Optional: OpenAI for additional AI features