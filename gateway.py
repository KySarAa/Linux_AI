from flask import Flask, request, Response
import requests

UPSTREAM = "http://lycee-malraux.fr:5656"
TOKEN = "5|In6J9lfn5IgkEQUuwhcWXCyuYNKVcJaYGbD6wyV726fc7975"
print("[GATEWAY] Token loaded:", TOKEN[:20] + "...")

app = Flask(__name__)

@app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy(path):
    url = f"{UPSTREAM}/api/{path}"

    headers = dict(request.headers)
    headers.pop("Host", None)
    headers["Accept"] = "application/json"
    headers["Authorization"] = f"Bearer {TOKEN}"

    resp = requests.request(
        method=request.method,
        url=url,
        params=request.args,
        data=request.get_data(),
        headers=headers,
        timeout=15,
    )

    return Response(resp.content, status=resp.status_code,
                    content_type=resp.headers.get("Content-Type"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8088)