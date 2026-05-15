import http.server
import socketserver
import os

PORT = 8000
DIRECTORY = "web"

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving the '{DIRECTORY}' directory at http://0.0.0.0:{PORT}")
    httpd.serve_forever()
