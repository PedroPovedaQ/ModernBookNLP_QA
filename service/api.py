import hashlib
import hmac

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

from service.config import Settings
from service.store import BusyJob, QueueFull, Store


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class BodyLimit:
    """Enforce the limit before JSON parsing, including chunked requests."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        messages, total = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            total += len(message.get("body", b""))
            if total > self.limit:
                return await JSONResponse(
                    {"detail": "Request too large"}, status_code=413
                )(scope, receive, send)
            messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay():
            return messages.pop(0) if messages else await receive()

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None):
    settings = settings or Settings.from_env()
    if not settings.keys:
        raise ValueError("Configure BOOKNLP_API_KEY_HASHES before starting the API")
    store = Store(settings)
    app = FastAPI(
        title="ModernBookNLP API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    # JSON escapes can expand a Unicode code point to 12 ASCII bytes.
    app.add_middleware(BodyLimit, limit=settings.max_chars * 12 + 1024)

    def owner(authorization: str | None = Header(default=None)):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                401, "Invalid credentials", headers={"WWW-Authenticate": "Bearer"}
            )
        digest = hashlib.sha256(authorization[7:].encode()).hexdigest()
        for expected, client in settings.keys.items():
            if hmac.compare_digest(digest, expected):
                return client
        raise HTTPException(
            401, "Invalid credentials", headers={"WWW-Authenticate": "Bearer"}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Default Pydantic errors may echo submitted private prose.
        return JSONResponse({"detail": "Invalid request body"}, status_code=422)

    @app.get("/healthz")
    def health():
        return {"status": "alive"}

    @app.get("/readyz")
    def ready():
        if not store.ready():
            raise HTTPException(503, "Worker unavailable or loading")
        return {"status": "ready"}

    @app.get("/v1/openapi.json")
    def schema(client: str = Depends(owner)):
        return app.openapi()

    @app.post("/v1/jobs", status_code=202)
    async def submit(body: Submission, client: str = Depends(owner)):
        if len(body.text) > settings.max_chars:
            raise HTTPException(
                413, f"Text exceeds {settings.max_chars} Unicode code points"
            )
        if not body.text.strip() or any(
            0xD800 <= ord(c) <= 0xDFFF or c == "\x00" for c in body.text
        ):
            raise HTTPException(422, "Text must be nonempty valid Unicode without NUL")
        try:
            job = await run_in_threadpool(store.submit, client, body.text)
        except QueueFull:
            raise HTTPException(
                429,
                "Queue or retention capacity reached",
                headers={"Retry-After": "30"},
            )
        return {"data": job}

    @app.get("/v1/jobs/{job_id}")
    def get(job_id: str, client: str = Depends(owner)):
        job = store.get(job_id, client)
        if job is None:
            raise HTTPException(404, "Job not found")
        return {"data": job}

    @app.delete("/v1/jobs/{job_id}", status_code=204)
    def delete(job_id: str, client: str = Depends(owner)):
        try:
            deleted = store.delete(job_id, client)
        except BusyJob:
            raise HTTPException(409, "Cannot delete a running job; wait for completion")
        if not deleted:
            raise HTTPException(404, "Job not found")
        return Response(status_code=204)

    return app
