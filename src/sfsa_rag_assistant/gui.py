"""
Browser-based beta UI for the SFSA Agentic RAG assistant.
"""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Tuple

from fastapi import FastAPI, Form, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, RedirectResponse
import uvicorn

from .app_config import settings
from .auth import (
    SESSION_COOKIE_NAME,
    SESSION_TIMEOUT_HOURS,
    create_session,
    get_authenticated_user_from_session,
    has_active_users,
    init_auth_db,
    invalidate_session,
    invalidate_user_sessions,
    log_usage,
    peek_authenticated_user_from_session,
    verify_credentials,
)
from .batch_processor import process_batch
from .config import check_ollama_connection

try:
    import gradio as gr
except ImportError:  # pragma: no cover - exercised only when dependency missing
    gr = None


def format_elapsed_time(seconds: float) -> str:
    """
    Format a duration in seconds for UI display.
    """
    total_seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def make_unique_output_path(output_filename: str) -> Path:
    """
    Resolve a writable output path without allowing overwrite collisions.
    """
    output_dir = settings.get_output_dir()
    raw_name = (output_filename or "").strip() or "sfsa_batch_results.csv"
    raw_name = raw_name if raw_name.lower().endswith(".csv") else f"{raw_name}.csv"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_name).strip("._") or "sfsa_batch_results.csv"

    candidate = output_dir / safe_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_candidate = output_dir / f"{stem}_{timestamp}{suffix}"

    counter = 1
    while unique_candidate.exists():
        unique_candidate = output_dir / f"{stem}_{timestamp}_{counter}{suffix}"
        counter += 1

    return unique_candidate


def run_batch_query(
    uploaded_csv,
    output_filename: str,
    progress=None,
    request: gr.Request | None = None,
) -> Tuple[str, str, str]:
    """
    Run the beta batch workflow and return summary, elapsed time, and output file path.
    """
    if not uploaded_csv:
        raise gr.Error("Please upload a CSV file before starting the batch run.")

    if progress is None and gr is not None:
        progress = gr.Progress()

    username = getattr(request, "username", None)
    source_ip = getattr(getattr(request, "client", None), "host", "")
    session_hash = getattr(request, "session_hash", "")
    output_csv = make_unique_output_path(output_filename)
    uploaded_csv_path, uploaded_name = normalize_uploaded_csv(uploaded_csv)

    started = perf_counter()
    try:
        if not check_ollama_connection():
            raise gr.Error(
                f"Cannot connect to Ollama at {settings.ollama_base_url} with model '{settings.ollama_model}'."
            )

        stats = process_batch(
            input_csv=uploaded_csv_path,
            output_csv=str(output_csv),
            question_column=None,
            max_validation_attempts=settings.max_validation_attempts,
            progress_callback=(
                (lambda completed, total, desc: progress(completed / total if total else 0, desc=desc))
                if progress is not None else None
            ),
        )
    except Exception:
        elapsed = perf_counter() - started
        log_usage(
            username=username or "",
            action="batch_query",
            status="error",
            model_name=settings.ollama_model,
            input_summary=uploaded_name,
            output_path=str(output_csv),
            source_ip=source_ip,
            session_hash=session_hash,
            metadata={
                "validation_attempts": settings.max_validation_attempts,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
        raise

    elapsed = perf_counter() - started
    elapsed_display = format_elapsed_time(elapsed)

    log_usage(
        username=username or "",
        action="batch_query",
        status="success",
        model_name=settings.ollama_model,
        input_summary=uploaded_name,
        output_path=str(output_csv),
        source_ip=source_ip,
        session_hash=session_hash,
        metadata={
            "total": stats["total"],
            "processed": stats["processed"],
            "successful": stats["successful"],
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "question_column": stats.get("question_column", ""),
            "validation_attempts": settings.max_validation_attempts,
            "elapsed_seconds": round(elapsed, 2),
        },
    )

    summary = "\n".join(
        [
            f"Output file: {output_csv.name}",
            f"Question column detected: {stats.get('question_column', 'N/A')}",
            f"Total questions: {stats['total']}",
            f"Processed: {stats['processed']}",
            f"Successful: {stats['successful']}",
            f"Failed: {stats['failed']}",
            f"Skipped: {stats['skipped']}",
            f"Total time: {elapsed_display}",
        ]
    )
    return summary, elapsed_display, str(output_csv)


def create_app():
    """
    Build and return the beta Gradio interface.
    """
    if gr is None:
        raise ImportError("Gradio is not installed. Install it with `pip install gradio`.")

    with gr.Blocks(title="SFSA Agentic RAG Assistant Beta") as app:
        gr.Markdown(
            f"""
            # SFSA Agentic RAG Assistant Beta
            Upload a CSV of questions to run the batch workflow.

            This beta uses the server's default model: `{settings.ollama_model}`.
            Output files are saved automatically in `{settings.get_output_dir()}`.

            [Log out](/logout)
            """
        )

        uploaded_csv = gr.File(
            label="Input CSV",
            file_types=[".csv"],
            type="binary",
            file_count="single",
            height=160,
        )
        output_filename = gr.Textbox(
            label="Output CSV filename",
            value="sfsa_batch_results.csv",
            placeholder="sfsa_batch_results.csv",
        )
        batch_submit = gr.Button("Run batch", variant="primary")
        batch_summary = gr.Textbox(label="Batch summary", lines=10)
        elapsed_time = gr.Textbox(label="Total processing time")
        batch_output_file = gr.File(label="Output CSV")

        batch_submit.click(
            fn=run_batch_query,
            inputs=[uploaded_csv, output_filename],
            outputs=[batch_summary, elapsed_time, batch_output_file],
        )

    return app


def login_page_html(error_message: str = "") -> str:
    """
    Render the custom login page.
    """
    error_block = (
        f'<p style="color:#b91c1c;margin-top:0.75rem;font-weight:600;">{error_message}</p>'
        if error_message else ""
    )
    return f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>SFSA RAG Login</title>
        <style>
          body {{
            margin: 0;
            font-family: Georgia, serif;
            background: linear-gradient(135deg, #f4efe4, #e6dcc7);
            min-height: 100vh;
            display: grid;
            place-items: center;
            color: #1f2937;
          }}
          .card {{
            width: min(420px, 92vw);
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(31,41,55,0.12);
            border-radius: 18px;
            padding: 2rem;
            box-shadow: 0 24px 60px rgba(31,41,55,0.16);
          }}
          h1 {{
            margin-top: 0;
            margin-bottom: 0.5rem;
            font-size: 1.9rem;
          }}
          p {{
            line-height: 1.5;
          }}
          label {{
            display: block;
            margin-top: 1rem;
            margin-bottom: 0.35rem;
            font-weight: 700;
          }}
          input {{
            width: 100%;
            box-sizing: border-box;
            padding: 0.85rem 0.95rem;
            border-radius: 10px;
            border: 1px solid #c9bca5;
            font-size: 1rem;
          }}
          button {{
            width: 100%;
            margin-top: 1.2rem;
            padding: 0.9rem 1rem;
            border: none;
            border-radius: 999px;
            background: #7c2d12;
            color: white;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
          }}
          .meta {{
            margin-top: 1rem;
            font-size: 0.92rem;
            color: #4b5563;
          }}
        </style>
      </head>
      <body>
        <div class="card">
          <h1>SFSA Member Login</h1>
          <p>Use your approved SFSA beta credentials to access the batch testing tool.</p>
          <form method="post" action="/login">
            <label for="username">Username</label>
            <input id="username" name="username" autocomplete="username" required />
            <label for="password">Password</label>
            <input id="password" name="password" type="password" autocomplete="current-password" required />
            <button type="submit">Sign In</button>
          </form>
          {error_block}
          <p class="meta">Sessions expire after {SESSION_TIMEOUT_HOURS} hours of inactivity.</p>
        </div>
      </body>
    </html>
    """


def create_server_app() -> FastAPI:
    """
    Create the FastAPI wrapper that handles login sessions and serves the Gradio app.
    """
    init_auth_db()
    gradio_app = create_app()
    app = FastAPI(title="SFSA Agentic RAG Beta")

    @app.middleware("http")
    async def session_middleware(request: FastAPIRequest, call_next):
        auth_enabled = has_active_users()
        session_token = request.cookies.get(SESSION_COOKIE_NAME, "")

        if auth_enabled and request.url.path.startswith("/app"):
            username = get_authenticated_user_from_session(session_token)
            if not username:
                return RedirectResponse(url="/login", status_code=303)
            response = await call_next(request)
            response.set_cookie(
                SESSION_COOKIE_NAME,
                session_token,
                max_age=SESSION_TIMEOUT_HOURS * 3600,
                httponly=True,
                samesite="lax",
                path="/",
            )
            return response

        return await call_next(request)

    @app.get("/", include_in_schema=False)
    async def root(request: FastAPIRequest):
        if has_active_users():
            username = get_authenticated_user_from_session(request.cookies.get(SESSION_COOKIE_NAME, ""))
            return RedirectResponse(url="/app" if username else "/login", status_code=303)
        return RedirectResponse(url="/app", status_code=303)

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False)
    async def login_form(request: FastAPIRequest):
        if not has_active_users():
            return RedirectResponse(url="/app", status_code=303)
        if request.query_params.get("logged_out") == "1":
            response = HTMLResponse(login_page_html("You have been logged out."))
            response.delete_cookie(SESSION_COOKIE_NAME, path="/")
            response.delete_cookie(SESSION_COOKIE_NAME, path="/app")
            return response
        if get_authenticated_user_from_session(request.cookies.get(SESSION_COOKIE_NAME, "")):
            return RedirectResponse(url="/app", status_code=303)
        return HTMLResponse(login_page_html())

    @app.post("/login", response_class=HTMLResponse, include_in_schema=False)
    async def login_submit(
        request: FastAPIRequest,
        username: str = Form(...),
        password: str = Form(...),
    ):
        if not verify_credentials(username, password):
            return HTMLResponse(login_page_html("Incorrect username or password."), status_code=401)

        response = RedirectResponse(url="/app", status_code=303)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            create_session(username),
            max_age=SESSION_TIMEOUT_HOURS * 3600,
            httponly=True,
            samesite="lax",
            path="/",
        )
        return response

    @app.get("/logout", include_in_schema=False)
    async def logout(request: FastAPIRequest):
        session_token = request.cookies.get(SESSION_COOKIE_NAME, "")
        username = peek_authenticated_user_from_session(session_token)
        invalidate_session(session_token)
        if username:
            invalidate_user_sessions(username)
        response = RedirectResponse(url="/login?logged_out=1", status_code=303)
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        response.delete_cookie(SESSION_COOKIE_NAME, path="/app")
        return response

    def auth_dependency(request: FastAPIRequest):
        if not has_active_users():
            return "anonymous"
        return get_authenticated_user_from_session(request.cookies.get(SESSION_COOKIE_NAME, ""))

    gr.mount_gradio_app(
        app,
        gradio_app,
        path="/app",
        auth_dependency=auth_dependency if has_active_users() else None,
        allowed_paths=[
            str(Path.cwd().resolve()),
            str(Path.home().resolve()),
        ],
    )
    return app


def launch_gui(host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> None:
    """
    Launch the beta UI server.
    """
    if share:
        raise ValueError("`share=True` is not supported for the FastAPI-mounted beta UI.")
    uvicorn.run(create_server_app(), host=host, port=port)


def normalize_uploaded_csv(uploaded_csv) -> Tuple[str, str]:
    """
    Normalize a Gradio upload into a server-side CSV path.
    """
    if isinstance(uploaded_csv, str):
        return uploaded_csv, Path(uploaded_csv).name

    if isinstance(uploaded_csv, bytes):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_file.write(uploaded_csv)
        temp_file.flush()
        temp_file.close()
        return temp_file.name, "uploaded_batch.csv"

    if hasattr(uploaded_csv, "read"):
        content = uploaded_csv.read()
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        temp_file.write(content)
        temp_file.flush()
        temp_file.close()
        name = getattr(uploaded_csv, "name", "uploaded_batch.csv")
        return temp_file.name, Path(name).name

    raise gr.Error("Uploaded CSV format was not recognized.")


def main() -> None:
    """
    Module entry point for `python -m sfsa_rag_assistant.gui`.
    """
    launch_gui()


if __name__ == "__main__":
    main()
