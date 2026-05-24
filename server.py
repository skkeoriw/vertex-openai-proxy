#!/usr/bin/env python3
"""
Vertex AI → OpenAI Compatible Proxy
支持 AQ.xxx 格式的 Vertex AI API Key，对外暴露完整 OpenAI API 格式。
"""
import json
import os
import time
import urllib.request
import urllib.error
import socket
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse


class ReusableHTTPServer(HTTPServer):
    allow_reuse_address = True
    allow_reuse_port = True

VERTEX_KEY        = os.environ.get("VERTEX_API_KEY", "")
MODEL             = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-lite")
PORT              = int(os.environ.get("PORT", "8765"))
MASTER_KEY        = os.environ.get("MASTER_KEY", "")
SYS_MAX_CHARS     = int(os.environ.get("SYS_MAX_CHARS", "3000"))  # 系统消息最大长度

VERTEX_BASE  = "https://aiplatform.googleapis.com/v1/publishers/google/models"


# ── 格式转换：OpenAI → Vertex AI ─────────────────────────────────────────────

def convert_request(body: dict) -> dict:
    messages  = body.get("messages", [])
    contents  = []
    sys_parts = []

    # 先收集所有消息，把连续 tool 响应合并到同一个 user message
    # Gemini 要求：同一 assistant turn 的多个 functionCall 必须对应
    # 同一个 user message 里的多个 functionResponse
    pending_tool_parts: list = []

    def flush_tool_parts():
        if pending_tool_parts:
            contents.append({"role": "user", "parts": list(pending_tool_parts)})
            pending_tool_parts.clear()

    for msg in messages:
        role           = msg.get("role", "user")
        content        = msg.get("content", "")
        tool_calls     = msg.get("tool_calls", [])
        parts          = []

        if role == "system":
            text = content if isinstance(content, str) else \
                   "".join(c.get("text","") for c in content if isinstance(c,dict))
            if len(text) > SYS_MAX_CHARS:
                text = text[:SYS_MAX_CHARS]
            sys_parts.append({"text": text})
            continue

        # tool 结果回传：合并连续多个到同一 user message
        if role == "tool":
            pending_tool_parts.append({"functionResponse": {
                "name": msg.get("name", "tool"),
                "response": {"output": content if isinstance(content, str)
                             else json.dumps(content)},
            }})
            continue

        # 非 tool 消息：先把积累的 tool responses flush 出去
        flush_tool_parts()

        # 普通文字内容
        if isinstance(content, str) and content:
            parts.append({"text": content})
        elif isinstance(content, list):
            for c in content:
                if isinstance(c, dict):
                    if c.get("type") == "text":
                        parts.append({"text": c["text"]})
                    elif c.get("type") == "image_url":
                        url = c.get("image_url", {}).get("url", "")
                        if url.startswith("data:"):
                            mime, data = url.split(";base64,")
                            mime = mime.replace("data:", "")
                            parts.append({"inlineData": {"mimeType": mime, "data": data}})

        # assistant 的 tool calls
        for tc in tool_calls:
            func = tc.get("function", {})
            args = func.get("arguments", "{}")
            if isinstance(args, str):
                try:    args = json.loads(args)
                except: args = {}
            parts.append({"functionCall": {
                "name": func.get("name", ""),
                "args": args,
            }})

        vertex_role = "model" if role == "assistant" else "user"
        if parts:
            contents.append({"role": vertex_role, "parts": parts})

    # 末尾可能还有未 flush 的 tool parts
    flush_tool_parts()

    req: dict = {"contents": contents}

    if sys_parts:
        req["systemInstruction"] = {"parts": sys_parts}

    # tools / function calling
    tools = body.get("tools", [])
    if tools:
        decls = []
        for t in tools:
            if t.get("type") == "function":
                f = t["function"]
                decls.append({
                    "name":        f.get("name", ""),
                    "description": f.get("description", ""),
                    "parameters":  f.get("parameters", {}),
                })
        req["tools"] = [{"functionDeclarations": decls}]

    # tool_choice
    choice = body.get("tool_choice")
    if choice == "required":
        req["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
    elif choice == "none":
        req["toolConfig"] = {"functionCallingConfig": {"mode": "NONE"}}
    elif isinstance(choice, dict) and choice.get("type") == "function":
        req["toolConfig"] = {"functionCallingConfig": {
            "mode": "ANY",
            "allowedFunctionNames": [choice["function"]["name"]],
        }}

    # generation config
    cfg = {}
    if "max_tokens"   in body: cfg["maxOutputTokens"] = body["max_tokens"]
    if "temperature"  in body: cfg["temperature"]     = body["temperature"]
    if "top_p"        in body: cfg["topP"]            = body["top_p"]
    if "stop"         in body:
        cfg["stopSequences"] = body["stop"] if isinstance(body["stop"], list) \
                               else [body["stop"]]
    if cfg:
        req["generationConfig"] = cfg

    return req


# ── 格式转换：Vertex AI → OpenAI ─────────────────────────────────────────────

def convert_response(vertex: dict) -> dict:
    candidates = vertex.get("candidates", [])
    message     = {"role": "assistant", "content": None}
    finish      = "stop"

    if candidates:
        cand   = candidates[0]
        parts  = cand.get("content", {}).get("parts", [])
        texts  = []
        tcalls = []

        for i, p in enumerate(parts):
            if "text" in p:
                texts.append(p["text"])
            elif "functionCall" in p:
                fc = p["functionCall"]
                tcalls.append({
                    "id":       f"call_{i}_{int(time.time())}",
                    "type":     "function",
                    "function": {
                        "name":      fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                })

        if texts:
            message["content"] = "".join(texts)
        if tcalls:
            message["tool_calls"] = tcalls
            finish = "tool_calls"

        fr_map = {"STOP": "stop", "MAX_TOKENS": "length",
                  "SAFETY": "content_filter", "RECITATION": "content_filter"}
        finish = fr_map.get(cand.get("finishReason", "STOP"), "stop")

    usage = vertex.get("usageMetadata", {})
    return {
        "id":      f"chatcmpl-{int(time.time())}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   MODEL,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage":   {
            "prompt_tokens":     usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens":      usage.get("totalTokenCount", 0),
        },
    }


def convert_stream_chunk(chunk: dict, cid: str) -> str:
    """单个 streaming chunk → OpenAI SSE 格式"""
    candidates = chunk.get("candidates", [])
    delta      = {}
    finish     = None

    if candidates:
        cand  = candidates[0]
        parts = cand.get("content", {}).get("parts", [])
        texts = []
        tcalls = []
        for i, p in enumerate(parts):
            if "text" in p:
                texts.append(p["text"])
            elif "functionCall" in p:
                fc = p["functionCall"]
                tcalls.append({
                    "index":    i,
                    "id":       f"call_{i}",
                    "type":     "function",
                    "function": {"name": fc.get("name",""),
                                 "arguments": json.dumps(fc.get("args",""))},
                })
        if texts:
            delta["content"] = "".join(texts)
        if tcalls:
            delta["tool_calls"] = tcalls

        fr = cand.get("finishReason")
        if fr and fr != "FINISH_REASON_UNSPECIFIED":
            fr_map = {"STOP": "stop", "MAX_TOKENS": "length"}
            finish = fr_map.get(fr, "stop")

    obj = {
        "id":      cid,
        "object":  "chat.completion.chunk",
        "created": int(time.time()),
        "model":   MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(obj)}\n\n"


# ── HTTP 处理 ─────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"[proxy] {self.command} {self.path} — {fmt % args}")

    def _auth_ok(self) -> bool:
        if not MASTER_KEY:
            return True
        auth = self.headers.get("Authorization", "")
        return auth.replace("Bearer ", "").strip() == MASTER_KEY

    def _send_json(self, status: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):
        if not self._auth_ok():
            self._send_json(401, {"error": {"message": "Unauthorized"}})
            return
        # /v1/models
        self._send_json(200, {"object": "list", "data": [
            {"id": MODEL, "object": "model", "owned_by": "google",
             "created": 1700000000},
        ]})

    def do_POST(self):
        if not self._auth_ok():
            self._send_json(401, {"error": {"message": "Unauthorized"}})
            return

        length  = int(self.headers.get("Content-Length", 0))
        body    = json.loads(self.rfile.read(length) or b"{}")
        stream  = body.get("stream", False)
        model   = body.get("model", MODEL) or MODEL

        vertex_body = convert_request(body)
        payload     = json.dumps(vertex_body).encode()

        endpoint = "streamGenerateContent" if stream else "generateContent"
        url = f"{VERTEX_BASE}/{model}:{endpoint}?key={VERTEX_KEY}"
        if stream:
            url += "&alt=sse"

        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"}
        )

        try:
            resp = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            err = json.loads(e.read() or b"{}")
            self._send_json(e.code, {"error": {
                "message": err.get("error", {}).get("message", str(e)),
                "type":    "api_error",
                "code":    e.code,
            }})
            return

        if stream:
            # 先收集所有 chunks，再一次性返回（避免 chunked encoding 兼容性问题）
            cid    = f"chatcmpl-{int(time.time())}"
            chunks = []
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_str)
                    chunks.append(convert_stream_chunk(chunk, cid).encode())
                except Exception:
                    pass
            chunks.append(b"data: [DONE]\n\n")
            body_bytes = b"".join(chunks)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body_bytes)))
            self.end_headers()
            self.wfile.write(body_bytes)
            self.wfile.flush()
        else:
            vertex_resp = json.loads(resp.read())
            self._send_json(200, convert_response(vertex_resp))


if __name__ == "__main__":
    if not VERTEX_KEY:
        print("❌ VERTEX_API_KEY 未设置")
        exit(1)
    print(f"[proxy] Vertex AI → OpenAI Proxy")
    print(f"[proxy] 监听: http://0.0.0.0:{PORT}")
    print(f"[proxy] 模型: {MODEL}")
    print(f"[proxy] 鉴权: {'MASTER_KEY 已设置' if MASTER_KEY else '无限制（建议设置 MASTER_KEY）'}")
    server = ReusableHTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
