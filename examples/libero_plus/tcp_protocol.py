"""Length-prefixed TCP transport for Motus LIBERO-plus evaluation."""

from __future__ import annotations

import base64
import json
import socket
import threading
import traceback
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(value.tobytes()).decode("ascii"),
                "dtype": str(value.dtype),
                "shape": value.shape,
            }
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        return super().default(value)


def _object_hook(value: dict) -> Any:
    if value.get("__numpy_array__"):
        raw = base64.b64decode(value["data"])
        return np.frombuffer(raw, dtype=value["dtype"]).reshape(value["shape"])
    return value


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ConnectionError("Connection closed while receiving a message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(connection: socket.socket, value: Any) -> None:
    payload = json.dumps(value, cls=NumpyEncoder).encode("utf-8")
    connection.sendall(len(payload).to_bytes(4, "big") + payload)


def receive_message(connection: socket.socket) -> Any:
    size = int.from_bytes(_recv_exact(connection, 4), "big")
    return json.loads(_recv_exact(connection, size).decode("utf-8"), object_hook=_object_hook)


class PolicyClient:
    def __init__(self, host: str, port: int, timeout: float = 300.0) -> None:
        self.connection = socket.create_connection((host, int(port)), timeout=timeout)

    def request(self, command: str, payload: Any = None) -> Any:
        send_message(self.connection, {"cmd": command, "payload": payload})
        response = receive_message(self.connection)
        if "error" in response:
            raise RuntimeError(f"Remote policy error: {response['error']}\n{response.get('traceback', '')}")
        return response["result"]

    def close(self) -> None:
        self.connection.close()


class PolicyServer:
    def __init__(self, policy: Any, host: str, port: int) -> None:
        self.policy = policy
        self.host = host
        self.port = int(port)

    def _handle(self, connection: socket.socket) -> None:
        with connection:
            while True:
                try:
                    request = receive_message(connection)
                    if request["cmd"] not in {"metadata", "predict_action"}:
                        raise ValueError("Unsupported policy command")
                    method = getattr(self.policy, request["cmd"])
                    payload = request.get("payload")
                    result = method(payload) if payload is not None else method()
                    send_message(connection, {"result": result})
                except ConnectionError:
                    return
                except Exception as error:
                    send_message(connection, {
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    })
                    return

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen(4)
            print(f"Motus LIBERO-plus server listening on {self.host}:{self.port}", flush=True)
            while True:
                connection, _ = server.accept()
                threading.Thread(target=self._handle, args=(connection,), daemon=True).start()
