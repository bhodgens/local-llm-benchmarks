#!/usr/bin/env python3
"""
Proxy for HumanEval chat completions.
Wraps raw code prompts with proper instructions and disables thinking.
Sits between lm-eval and llama.cpp server.
"""
import json
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

TARGET = "http://127.0.0.1:18099"
PROXY_PORT = 18098

SYSTEM_PROMPT = (
    "You are a Python code completion assistant. "
    "Complete the given Python function. "
    "Output ONLY the function body and implementation code. "
    "Do not explain, do not add markdown formatting, do not add test code. "
    "Just output valid Python code that completes the function."
)

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)
        
        # Inject system prompt and disable thinking
        messages = data.get('messages', [])
        
        # Check if first message is already a system prompt
        has_system = any(m.get('role') == 'system' for m in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        
        data['messages'] = messages
        
        # Disable thinking for all models
        data['chat_template_kwargs'] = {'enable_thinking': False}
        
        # Increase max_tokens to give room for thinking + code
        if data.get('max_tokens', 0) < 8192:
            data['max_tokens'] = 8192
        
        # Forward to target
        url = TARGET + self.path
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        try:
            resp = urllib.request.urlopen(req, timeout=120)
            resp_data = resp.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(resp_data)
        except Exception as e:
            self.send_response(502)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_GET(self):
        url = TARGET + self.path
        req = urllib.request.Request(url)
        try:
            resp = urllib.request.urlopen(req, timeout=5)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(resp.read())
        except:
            self.send_response(502)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass  # Suppress logging

if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', PROXY_PORT), ProxyHandler)
    print(f"HumanEval proxy on port {PROXY_PORT} -> {TARGET}")
    server.serve_forever()
