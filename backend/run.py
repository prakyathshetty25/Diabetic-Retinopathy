"""
Backend application launcher for Universal Retinal Screening framework.
"""

import uvicorn
from app.config import HOST, PORT

if __name__ == "__main__":
    print(f"Starting Universal Retinal Screening API on http://{HOST}:{PORT}")
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
