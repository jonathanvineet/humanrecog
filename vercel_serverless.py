import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys
sys.path.insert(0, os.path.dirname(__file__))

# For Vercel serverless functions
from api import app

def handler(event, context):
    """Vercel serverless function handler"""
    return app(event, context)
