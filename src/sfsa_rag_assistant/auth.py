"""
Member authentication and usage auditing for the SFSA GUI.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .app_config import settings


PBKDF2_ITERATIONS = 600_000
SESSION_COOKIE_NAME = "sfsa_session"
SESSION_TIMEOUT_HOURS = 12


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: str) -> datetime:
    """Parse a stored UTC timestamp."""
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get_auth_db_path() -> Path:
    """Return the configured authentication database path."""
    return settings.get_auth_db_path()


def get_connection() -> sqlite3.Connection:
    """Open a SQLite connection for auth and audit data."""
    connection = sqlite3.connect(get_auth_db_path())
    connection.row_factory = sqlite3.Row
    return connection


def init_auth_db() -> None:
    """Create auth tables if they do not already exist."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT,
                email TEXT,
                role TEXT NOT NULL DEFAULT 'member',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT,
                last_used_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS usage_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                model_name TEXT,
                input_summary TEXT,
                output_path TEXT,
                source_ip TEXT,
                session_hash TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(username) REFERENCES users(username)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_activity_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                FOREIGN KEY(username) REFERENCES users(username)
            )
            """
        )


def hash_password(password: str, salt: str) -> str:
    """Hash a password using PBKDF2-HMAC-SHA256."""
    password_bytes = password.encode("utf-8")
    salt_bytes = bytes.fromhex(salt)
    derived = hashlib.pbkdf2_hmac("sha256", password_bytes, salt_bytes, PBKDF2_ITERATIONS)
    return derived.hex()


def upsert_user(
    username: str,
    password: str,
    *,
    full_name: str = "",
    email: str = "",
    role: str = "member",
    is_active: bool = True,
) -> None:
    """Create or update a member account."""
    init_auth_db()
    normalized_username = username.strip().lower()
    if not normalized_username:
        raise ValueError("Username cannot be empty.")
    if not password:
        raise ValueError("Password cannot be empty.")

    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    now = utc_now()

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO users (
                username, password_hash, salt, full_name, email, role,
                is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                password_hash = excluded.password_hash,
                salt = excluded.salt,
                full_name = excluded.full_name,
                email = excluded.email,
                role = excluded.role,
                is_active = excluded.is_active,
                updated_at = excluded.updated_at
            """,
            (
                normalized_username,
                password_hash,
                salt,
                full_name.strip(),
                email.strip(),
                role.strip() or "member",
                1 if is_active else 0,
                now,
                now,
            ),
        )
    invalidate_user_sessions(normalized_username)


def set_user_active(username: str, is_active: bool) -> bool:
    """Activate or deactivate a user."""
    init_auth_db()
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE users SET is_active = ?, updated_at = ? WHERE username = ?",
            (1 if is_active else 0, utc_now(), username.strip().lower()),
        )
        updated = cursor.rowcount > 0
    if updated and not is_active:
        invalidate_user_sessions(username)
    return updated


def has_active_users() -> bool:
    """Return True if at least one active account exists."""
    init_auth_db()
    with get_connection() as connection:
        row = connection.execute("SELECT COUNT(*) AS count FROM users WHERE is_active = 1").fetchone()
    return bool(row["count"])


def verify_credentials(username: str, password: str) -> bool:
    """Verify a username/password pair against the local user database."""
    init_auth_db()
    normalized_username = username.strip().lower()
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT username, password_hash, salt
            FROM users
            WHERE username = ? AND is_active = 1
            """,
            (normalized_username,),
        ).fetchone()

        if row is None:
            return False

        candidate_hash = hash_password(password, row["salt"])
        is_valid = hmac.compare_digest(candidate_hash, row["password_hash"])
        if is_valid:
            connection.execute(
                "UPDATE users SET last_login_at = ?, updated_at = ? WHERE username = ?",
                (utc_now(), utc_now(), normalized_username),
            )
        return is_valid


def log_usage(
    *,
    username: str,
    action: str,
    status: str,
    model_name: str = "",
    input_summary: str = "",
    output_path: str = "",
    source_ip: str = "",
    session_hash: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a member action in the usage audit log."""
    if not username:
        return

    init_auth_db()
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO usage_logs (
                username, action, status, model_name, input_summary, output_path,
                source_ip, session_hash, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username.strip().lower(),
                action,
                status,
                model_name,
                input_summary[:2000],
                output_path,
                source_ip,
                session_hash,
                json.dumps(metadata or {}, ensure_ascii=True),
                utc_now(),
            ),
        )
        connection.execute(
            "UPDATE users SET last_used_at = ?, updated_at = ? WHERE username = ?",
            (utc_now(), utc_now(), username.strip().lower()),
        )


def list_users() -> list[sqlite3.Row]:
    """Return all users for admin inspection."""
    init_auth_db()
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT username, full_name, email, role, is_active,
                   created_at, last_login_at, last_used_at
            FROM users
            ORDER BY username
            """
        ).fetchall()


def list_usage(limit: int = 50) -> list[sqlite3.Row]:
    """Return recent usage records."""
    init_auth_db()
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT username, action, status, model_name, input_summary,
                   output_path, source_ip, session_hash, created_at
            FROM usage_logs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()


def auth_callback(username: str, password: str) -> bool:
    """Gradio auth callback."""
    return verify_credentials(username, password)


def create_session(username: str) -> str:
    """Create a persistent login session for a user."""
    init_auth_db()
    normalized_username = username.strip().lower()
    token = secrets.token_urlsafe(32)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(hours=SESSION_TIMEOUT_HOURS)

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO sessions (
                session_token, username, created_at, last_activity_at, expires_at, is_active
            ) VALUES (?, ?, ?, ?, ?, 1)
            """,
            (
                token,
                normalized_username,
                now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                expires_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            ),
        )
    return token


def invalidate_session(session_token: str) -> None:
    """Deactivate a session token."""
    if not session_token:
        return
    init_auth_db()
    with get_connection() as connection:
        connection.execute(
            "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
            (session_token,),
        )


def invalidate_user_sessions(username: str) -> None:
    """Deactivate all sessions for a user."""
    init_auth_db()
    with get_connection() as connection:
        connection.execute(
            "UPDATE sessions SET is_active = 0 WHERE username = ?",
            (username.strip().lower(),),
        )


def get_authenticated_user_from_session(session_token: str) -> Optional[str]:
    """Validate a session token and refresh its sliding expiration."""
    if not session_token:
        return None

    init_auth_db()
    now = datetime.now(timezone.utc)
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT s.username, s.expires_at, u.is_active
            FROM sessions s
            JOIN users u ON u.username = s.username
            WHERE s.session_token = ? AND s.is_active = 1
            """,
            (session_token,),
        ).fetchone()

        if row is None:
            return None

        if not row["is_active"]:
            connection.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
                (session_token,),
            )
            return None

        expires_at = parse_utc_timestamp(row["expires_at"])
        if expires_at <= now:
            connection.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
                (session_token,),
            )
            return None

        refreshed_expires = now + timedelta(hours=SESSION_TIMEOUT_HOURS)
        connection.execute(
            """
            UPDATE sessions
            SET last_activity_at = ?, expires_at = ?
            WHERE session_token = ?
            """,
            (
                now.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                refreshed_expires.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                session_token,
            ),
        )
        return row["username"]


def peek_authenticated_user_from_session(session_token: str) -> Optional[str]:
    """Validate a session token without refreshing its expiration."""
    if not session_token:
        return None

    init_auth_db()
    now = datetime.now(timezone.utc)
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT s.username, s.expires_at, u.is_active
            FROM sessions s
            JOIN users u ON u.username = s.username
            WHERE s.session_token = ? AND s.is_active = 1
            """,
            (session_token,),
        ).fetchone()

        if row is None or not row["is_active"]:
            return None

        expires_at = parse_utc_timestamp(row["expires_at"])
        if expires_at <= now:
            connection.execute(
                "UPDATE sessions SET is_active = 0 WHERE session_token = ?",
                (session_token,),
            )
            return None

        return row["username"]


def print_users() -> None:
    """Print a simple user table."""
    users = list_users()
    if not users:
        print("No users found.")
        return
    for user in users:
        status = "active" if user["is_active"] else "inactive"
        print(
            f"{user['username']}\t{status}\t{user['role']}\t"
            f"{user['full_name'] or ''}\t{user['email'] or ''}\t"
            f"last_login={user['last_login_at'] or 'never'}\t"
            f"last_used={user['last_used_at'] or 'never'}"
        )


def print_usage(limit: int) -> None:
    """Print recent usage records."""
    rows = list_usage(limit=limit)
    if not rows:
        print("No usage records found.")
        return
    for row in rows:
        print(
            f"{row['created_at']}\t{row['username']}\t{row['action']}\t{row['status']}\t"
            f"model={row['model_name'] or '-'}\tip={row['source_ip'] or '-'}\t"
            f"output={row['output_path'] or '-'}\tinput={row['input_summary'][:120]}"
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the auth admin CLI parser."""
    parser = argparse.ArgumentParser(description="Manage SFSA GUI member accounts and usage logs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_user = subparsers.add_parser("create-user", help="Create or update a member account")
    create_user.add_argument("username", help="Username for the member")
    create_user.add_argument("--full-name", default="", help="Member full name")
    create_user.add_argument("--email", default="", help="Member email address")
    create_user.add_argument("--role", default="member", help="Role label to store for the member")

    deactivate_user = subparsers.add_parser("deactivate-user", help="Deactivate a member account")
    deactivate_user.add_argument("username", help="Username to deactivate")

    activate_user = subparsers.add_parser("activate-user", help="Reactivate a member account")
    activate_user.add_argument("username", help="Username to reactivate")

    subparsers.add_parser("list-users", help="List all member accounts")

    usage = subparsers.add_parser("usage", help="Show recent usage logs")
    usage.add_argument("--limit", type=int, default=50, help="Number of recent records to show")

    return parser


def main() -> None:
    """Admin CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()
    init_auth_db()

    if args.command == "create-user":
        password = getpass.getpass(f"Password for {args.username}: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            raise SystemExit("Passwords did not match.")
        upsert_user(
            args.username,
            password,
            full_name=args.full_name,
            email=args.email,
            role=args.role,
        )
        print(f"User '{args.username.lower()}' created or updated.")
        print(f"Auth DB: {get_auth_db_path()}")
        return

    if args.command == "deactivate-user":
        updated = set_user_active(args.username, False)
        print(f"User '{args.username.lower()}' deactivated." if updated else "User not found.")
        return

    if args.command == "activate-user":
        updated = set_user_active(args.username, True)
        print(f"User '{args.username.lower()}' activated." if updated else "User not found.")
        return

    if args.command == "list-users":
        print_users()
        return

    if args.command == "usage":
        print_usage(limit=max(1, args.limit))
        return


if __name__ == "__main__":
    main()
