#!/usr/bin/env python3
"""
Proxy that forwards OpenAI completions format to llama.cpp native /completion.
Works for code completion models. Adds system prompt wrapper for instruct models.
"""
import json
import sys
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler

LLAMA_HOST = "http://127.0.0.1:18099"

class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len)
        
        try:
            req = json.loads(body)
            prompt = req.get("prompt", "")
            
            # Try native completion first (works for code completion models)
            native_req = {
                "prompt": prompt,
                "n_predict": req.get("max_tokens", 1024),
                "temperature": req.get("temperature", 0.0),
                "stream": False,
            }
            
            if "stop" in req:
                stop = req["stop"]
                if isinstance(stop, str):
                    stop = [stop]
                native_req["stop"] = stop
            
            native_body = json.dumps(native_req).encode()
            native_request = urllib.request.Request(
                f"{LLAMA_HOST}/completion",
                data=native_body,
                headers={"Content-Type": "application/json"}
            )
            
            content = ""
            try:
                resp = urllib.request.urlopen(native_request, timeout=120)
                d = json.loads(resp.read())
                content = d.get("content", "")
                
                # If we got empty or garbage, try chat as fallback
                if len(content.strip()) < 2:
                    chat_req = {
                        "messages": [{"role": "user", "content": "Complete the following Python code. Output ONLY the code continuation, no explanations:\n\n" + prompt}],
                        "max_tokens": req.get("max_tokens", 1024),
                        "temperature": req.get("temperature", 0.0),
                        "stream": False,
                    }
                    if "stop" in req:
                        chat_req["stop"] = native_req.get("stop", [])
                    chat_body = json.dumps(chat_req).encode()
                    chat_request = urllib.request.Request(
                        f"{LLAMA_HOST}/v1/chat/completions",
                        data=chat_body,
                        headers={"Content-Type": "application/json"}
                    )
                    try:
                        chat_resp = urllib.request.urlopen(chat_request, timeout=120)
                        chat_d = json.loads(chat_resp.read())
                        chat_content = chat_d.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if len(chat_content.strip()) > 0:
                            content = chat_content
                    except:
                        pass
            except urllib.error.HTTPError:
                # Native endpoint failed (content validation), try chat
                try:
                    chat_req = {
                        "messages": [{"role": "user", "content": "Complete the following Python code. Output ONLY the code continuation:\n\n" + prompt}],
                        "max_tokens": req.get("max_tokens", 1024),
                        "temperature": req.get("temperature", 0.0),
                        "stream": False,
                    }
                    chat_body = json.dumps(chat_req).encode()
                    chat_request = urllib.request.Request(
                        f"{LLAMA_HOST}/v1/chat/completions",
                        data=chat_body,
                        headers={"Content-Type": "application/json"}
                    )
                    chat_resp = urllib.request.urlopen(chat_request, timeout=120)
                    chat_d = json.loads(chat_resp.read())
                    content = chat_d.get("choices", [{}])[0].get("message", {}).get("content", "")
                except:
                    content = ""
            except:
                content = ""
            
            openai_resp = {
                "id": "cmpl-proxy",
                "object": "text_completion",
                "created": 0,
                "model": req.get("model", "unknown"),
                "choices": [{"text": content, "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            
            resp_body = json.dumps(openai_resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
        except Exception:
            openai_resp = {
                "id": "cmpl-proxy-error",
                "object": "text_completion", "created": 0, "model": "unknown",
                "choices": [{"text": "", "finish_reason": "stop", "index": 0}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            resp_body = json.dumps(openai_resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)
    
    def do_GET(self):
        try:
            r = urllib.request.urlopen(f"{LLAMA_HOST}/health", timeout=3)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(r.read())
        except:
            self.send_response(503)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18098
    server = HTTPServer(("127.0.0.1", port), ProxyHandler)
    print(f"Smart proxy on 127.0.0.1:{port} -> {LLAMA_HOST}")
    server.serve_forever()
