"""Script to encode Telegram session file to base64 for deployment"""
import base64
import os

session_file = "telegram_session.session"

if not os.path.exists(session_file):
    print(f"ERROR: Session file '{session_file}' not found!")
    print(f"Make sure you've run the Telegram worker at least once to create the session file.")
    exit(1)

print(f"Reading session file: {session_file}")

try:
    with open(session_file, "rb") as f:
        session_data = f.read()
    
    # Encode to base64
    session_b64 = base64.b64encode(session_data).decode()
    
    # Save to file for easy copying
    output_file = "telegram_session_base64.txt"
    with open(output_file, "w") as f:
        f.write(session_b64)
    
    print(f"SUCCESS: Base64 encoded session saved to: {output_file}")
    print(f"File size: {len(session_b64)} characters")
    print(f"\nCopy the entire contents of '{output_file}' and paste it as the value for")
    print(f"TELEGRAM_SESSION_B64 in your Streamlit Cloud secrets.")
    print(f"\nFirst 100 characters: {session_b64[:100]}...")
    print(f"Last 100 characters: ...{session_b64[-100:]}")
    
except Exception as e:
    print(f"ERROR: {e}")

