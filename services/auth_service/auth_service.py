import os
import logging
import random
import string
import hashlib
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv

# Try to use the faster C implementation if available, otherwise fall back to Python
try:
    import mysql.connector
except ImportError:
    print("Error: MySQL Connector not installed. Please run 'pip install mysql-connector-python'")
    exit(1)

# ==============================================================================
# CONFIGURATION & UTILITIES
# ==============================================================================

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Application Instance
app = FastAPI(title="Authentication API Service")

# Database Connection Details from .env
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "database": os.getenv("DB_DATABASE", "authentication"),
}


# --- Utility Functions ---

def hash_password(password: str) -> str:
    """Returns the SHA256 hash of the input password."""
    print(f"DEBUG: Password received by hash_password: {password}")
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def generate_api_key() -> str:
    """Generates a 5-character key (3 uppercase letters + 2 digits)."""
    letters = ''.join(random.choices(string.ascii_uppercase, k=3))
    numbers = str(random.randint(10, 99))
    return letters + numbers


# --- Database Management ---

def execute_stored_procedure(procedure_name: str, args: tuple = ()) -> Optional[List[Dict[str, Any]]]:
    """Connects to MySQL, executes a stored procedure, and returns the result."""
    conn = None
    try:
        # 1. Establish connection
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor(dictionary=True)  # Return results as dictionaries (JSON-like)

        # 2. Execute procedure
        cursor.callproc(procedure_name, args)

        # 3. Handle results
        results = []
        for result in cursor.stored_results():
            results.extend(result.fetchall())

        # 4. Commit (only necessary for procedures that modify data, like REGISTER or REGENERATE)
        if procedure_name in ('sp_RegisterUser', 'sp_RegenerateApiKey', 'sp_LoginUser'):
            conn.commit()

        cursor.close()
        return results

    except mysql.connector.Error as err:
        logger.error(f"Database Error executing {procedure_name}: {err}")
        # Rollback any changes in case of error
        if conn:
            conn.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database error during {procedure_name} execution."
        )
    finally:
        if conn and conn.is_connected():
            conn.close()


# ==============================================================================
# Pydantic Models for Request/Response Validation
# ==============================================================================

class UserCredentials(BaseModel):
    username: str
    password: str


class APIKeyRequest(BaseModel):
    api_key: str


# ==============================================================================
# API ENDPOINTS (The 5 Routes)
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. /health - Used for basic service status check
# ------------------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
def get_health():
    """Checks if the service is running."""
    return {"status": "healthy"}


# ------------------------------------------------------------------------------
# 2. /register - Registers a new user
# ------------------------------------------------------------------------------
@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCredentials):
    """
    Registers a new user. Hashes the password and generates an API key
    before calling sp_RegisterUser.
    """
    password_hash = hash_password(user.password)
    new_api_key = generate_api_key()

    # Execute stored procedure: sp_RegisterUser(@username, @password_hash, @api_key)
    results = execute_stored_procedure("sp_RegisterUser", (user.username, password_hash, new_api_key))

    if not results:
        # This case shouldn't happen if sp_RegisterUser always returns success/error row
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Registration failed unexpectedly.")

    response = results[0]

    if response.get("status") == "error":
        # Handle "Username already exists" error returned by the SP
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=response.get("message", "Registration failed.")
        )

    # Success: returns status, username, and the new API key
    return response



# ------------------------------------------------------------------------------
# 3. /login - Authenticates and generates a NEW API key
# ------------------------------------------------------------------------------
@app.post("/login", status_code=status.HTTP_200_OK)
def login_user(user: UserCredentials):
    """
    Authenticates a user, generates a NEW API key, updates the database, 
    and returns the new key for session use.
    """
    password_hash = hash_password(user.password)
    
    # 1. GENERATE THE NEW KEY HERE
    new_api_key = generate_api_key() 
    
    # 2. EXECUTE SP with the NEW KEY as an argument
    results = execute_stored_procedure(
        "sp_LoginUser", 
        (user.username, password_hash,new_api_key)
    )

    if not results or results[0].get("status") == "error":
        # Handle "Invalid username or password" error returned by the SP
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=results[0].get("message", "Invalid username or password.")
        )
    
    # 3. SUCCESS: returns status, username, and the NEW api_key
    return results[0]


# ------------------------------------------------------------------------------
# 4. /validate-api-key - Validates an API key (used by Receiver Service)
# ------------------------------------------------------------------------------
@app.post("/validate-api-key", status_code=status.HTTP_200_OK)
def validate_api_key(key_req: APIKeyRequest):
    """
    Validates an API key by calling sp_ValidateApiKey.
    Returns the username if valid.
    """
    # Execute stored procedure: sp_ValidateApiKey(@api_key)
    results = execute_stored_procedure("sp_ValidateApiKey", (key_req.api_key,))

    if not results:
        # sp_ValidateApiKey returns no rows if the key is invalid
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key is invalid or expired."
        )

    # Success: returns {"valid": true, "username": "..."}
    return {"valid": True, "username": results[0]['username']}


# ------------------------------------------------------------------------------
# 5. /regenerate-api-key - Generates and updates a new API key
# ------------------------------------------------------------------------------
@app.post("/regenerate-api-key", status_code=status.HTTP_200_OK)
def regenerate_api_key(user: UserCredentials):
    """
    Authenticates the user, generates a new API key, and updates the database
    via sp_RegenerateApiKey.
    """
    password_hash = hash_password(user.password)
    new_api_key = generate_api_key()

    # Execute stored procedure: sp_RegenerateApiKey(@username, @password_hash, @new_api_key)
    results = execute_stored_procedure("sp_RegenerateApiKey", (user.username, password_hash, new_api_key))

    if not results:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Key regeneration failed unexpectedly.")

    response = results[0]

    if response.get("status") == "error":
        # Handle "Invalid credentials" error returned by the SP
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=response.get("message", "Invalid credentials.")
        )

    # Success: returns status and the new API key
    return response


# ==============================================================================
# RUN THE SERVICE
# ==============================================================================

if __name__ == "__main__":
    import uvicorn

    # The instructions require the service to run on port 8004
    logger.info("Starting Authentication Service on http://0.0.0.0:8004")
    # Add a visual aid for the setup
    #
    uvicorn.run(app, host="0.0.0.0", port=8004)