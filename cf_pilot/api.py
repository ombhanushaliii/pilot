"""Public HTTP transport with frozen response caching and bounded retries."""

import hashlib
import json
import random
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .store import canonical


class CollectionError(RuntimeError):
    pass


def request_key(url, params):
    return hashlib.sha256((url + "\n" + canonical(params)).encode()).hexdigest()


def http_get(url, timeout):
    with urlopen(Request(url, headers={"User-Agent": "CF-Pilot-CourseProject/0.1",
                                      "Accept": "application/json,text/html"}), timeout=timeout) as r:
        return r.read()


class Client:
    def __init__(self, store, run_id, interval=2.1, timeout=40, max_attempts=4,
                 transport=http_get, clock=time.time, sleep=time.sleep):
        if interval < 2.1 or timeout <= 0 or max_attempts < 1:
            raise ValueError("Unsafe or invalid HTTP settings")
        self.store, self.run_id = store, run_id
        self.interval, self.timeout, self.max_attempts = interval, timeout, max_attempts
        self.transport, self.clock, self.sleep = transport, clock, sleep
        self.network_requests = 0
        self.cache_hits = 0

    def _get(self, url, params, validator):
        key = request_key(url, params)
        cached = self.store.cached(self.run_id, key)
        if cached is not None:
            self.cache_hits += 1
            return validator(cached), key
        request_url = url + ("?" + urlencode(params) if params else "")
        for attempt in range(self.max_attempts):
            started = self.clock()
            try:
                with self.store.request_slot(self.interval, self.clock, self.sleep) as started:
                    self.network_requests += 1
                    body = self.transport(request_url, self.timeout)
                result = validator(body)
                self.store.archive(self.run_id, key, url, params, body)
                self._attempt(key, started, "OK", None)
                return result, key
            except (HTTPError, URLError, TimeoutError, OSError, CollectionError, ValueError) as e:
                self._attempt(key, started, "FAILED", str(e)[:500])
                temporary = isinstance(e, (URLError, TimeoutError, OSError))
                if isinstance(e, HTTPError):
                    temporary = e.code in {408, 429, 500, 502, 503, 504}
                if isinstance(e, CollectionError):
                    temporary = any(x in str(e).casefold() for x in ["limit exceeded", "temporarily", "try again", "unavailable"])
                if not temporary or attempt + 1 == self.max_attempts:
                    raise CollectionError(f"Request failed: {url} {params}: {e}") from e
                self.sleep(min(30.0, 2 ** (attempt + 1)) + random.random() * 0.25)
        raise AssertionError("Unreachable")

    def _attempt(self, key, started, outcome, detail):
        self.store.db.execute("INSERT INTO request_attempts(run_id,request_key,started_at,outcome,detail) VALUES(?,?,?,?,?)",
                              (self.run_id, key, started, outcome, detail))
        self.store.db.commit()

    def call(self, method, **params):
        def validate(body):
            try:
                document = json.loads(body)
            except (ValueError, UnicodeError) as e:
                raise CollectionError("Response is not valid JSON") from e
            if not isinstance(document, dict) or document.get("status") != "OK" or "result" not in document:
                comment = document.get("comment", "Malformed API response") if isinstance(document, dict) else "Malformed API response"
                raise CollectionError(str(comment))
            return document["result"]
        return self._get("https://codeforces.com/api/" + method, params, validate)

    def page(self, url):
        return self._get(url, {}, lambda body: body.decode("utf-8"))
