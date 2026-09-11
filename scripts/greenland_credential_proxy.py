"""Expose Greenland's refreshable role credentials to the main container."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


BIND_ADDRESS = "127.0.0.1"
BIND_PORT = int(os.environ.get("PROXY_BIND_PORT", "9090"))
REFRESH_BEFORE_SECONDS = 600


class CredentialCache:
    def __init__(self):
        self._lock = threading.Lock()
        self._credentials = None
        self._expiry = None
        self.refresh()

    def refresh(self):
        import boto3

        credentials = boto3.Session().get_credentials()
        if credentials is None:
            raise RuntimeError("Greenland credential profile returned no credentials")
        frozen = credentials.get_frozen_credentials()
        expiry = getattr(credentials, "_expiry_time", None)
        with self._lock:
            self._credentials = frozen
            self._expiry = expiry
        print(
            "[credential-proxy] credentials refreshed"
            f" expiry={expiry.isoformat() if expiry else 'unknown'}",
            flush=True,
        )

    def payload(self):
        with self._lock:
            credentials = self._credentials
            expiry = self._expiry
        if credentials is None:
            raise RuntimeError("Credential cache is empty")
        expiration = (
            expiry.astimezone(timezone.utc).isoformat()
            if expiry is not None
            else datetime.fromtimestamp(
                time.time() + 3600, timezone.utc
            ).isoformat()
        )
        return {
            "AccessKeyId": credentials.access_key,
            "SecretAccessKey": credentials.secret_key,
            "Token": credentials.token,
            "Expiration": expiration,
        }

    def seconds_until_refresh(self):
        with self._lock:
            expiry = self._expiry
        if expiry is None:
            return 2700
        return max(expiry.timestamp() - time.time() - REFRESH_BEFORE_SECONDS, 60)


def _refresh_loop(cache):
    while True:
        time.sleep(cache.seconds_until_refresh())
        try:
            cache.refresh()
        except Exception as error:
            print(
                f"[credential-proxy] refresh failed: {type(error).__name__}: "
                f"{error}",
                flush=True,
            )
            time.sleep(60)


def _main_container_watcher():
    own_pid = str(os.getpid())

    def other_pids():
        return [
            value
            for value in os.listdir("/proc")
            if value.isdigit() and value not in {own_pid, "1"}
        ]

    # Containers start concurrently. Do not mistake a main container that has
    # not entered the shared PID namespace yet for a completed process.
    deadline = time.time() + 600
    while time.time() < deadline:
        try:
            if other_pids():
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        print("[credential-proxy] main container did not start", flush=True)
        os._exit(1)

    while True:
        try:
            if not other_pids():
                print("[credential-proxy] main container exited", flush=True)
                os._exit(0)
        except OSError:
            pass
        time.sleep(5)


def _handler(cache, expected_token):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/creds":
                self.send_error(404)
                return
            if self.headers.get("Authorization") != expected_token:
                self.send_error(401)
                return
            payload = json.dumps(cache.payload()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, _format, *_args):
            return

    return Handler


def main():
    auth_token = os.environ.get("PROXY_AUTH_TOKEN")
    if not auth_token:
        raise RuntimeError("PROXY_AUTH_TOKEN is required")
    cache = CredentialCache()
    threading.Thread(
        target=_refresh_loop, args=(cache,), daemon=True
    ).start()
    threading.Thread(target=_main_container_watcher, daemon=True).start()
    server = ThreadingHTTPServer(
        (BIND_ADDRESS, BIND_PORT), _handler(cache, auth_token)
    )
    print(
        f"[credential-proxy] listening on {BIND_ADDRESS}:{BIND_PORT}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
