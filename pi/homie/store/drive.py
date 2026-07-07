"""Google Drive backed store — the production system of record.

Each document maps to one Drive file (list.txt / recipes.txt / memory.txt), shared
with the service account. Reachable from home (Pi) and remote (Telegram) alike, so
no channel owns the state.

The three files must be shared (Editor) with the service account email and their
file IDs provided via env: HOMIE_LIST_FILE_ID, HOMIE_RECIPES_FILE_ID,
HOMIE_MEMORY_FILE_ID.
"""

import io
import os

from homie.store import DOCS, Store

SCOPES = ["https://www.googleapis.com/auth/drive"]


class DriveStore(Store):
    def __init__(self, service_account_path: str, file_ids: dict[str, str]):
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        missing = [d for d in DOCS if not file_ids.get(d)]
        if missing:
            raise ValueError(f"DriveStore missing file IDs for: {missing}")

        creds = service_account.Credentials.from_service_account_file(
            service_account_path, scopes=SCOPES
        )
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)
        self._file_ids = file_ids

    @classmethod
    def from_env(cls) -> "DriveStore":
        from homie.config import REPO_DIR

        sa_path = os.environ.get(
            "GOOGLE_SERVICE_ACCOUNT_PATH",
            str(REPO_DIR / "secrets" / "google-calendar.json"),
        )
        file_ids = {
            "list": os.environ.get("HOMIE_LIST_FILE_ID", ""),
            "recipes": os.environ.get("HOMIE_RECIPES_FILE_ID", ""),
            "memory": os.environ.get("HOMIE_MEMORY_FILE_ID", ""),
        }
        return cls(sa_path, file_ids)

    def read(self, doc: str) -> str:
        from googleapiclient.http import MediaIoBaseDownload

        request = self._service.files().get_media(fileId=self._file_ids[doc])
        buffer = io.BytesIO()
        downloader = MediaIoBaseDownload(buffer, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buffer.getvalue().decode("utf-8")

    def write(self, doc: str, content: str) -> None:
        from googleapiclient.http import MediaIoBaseUpload

        media = MediaIoBaseUpload(
            io.BytesIO(content.encode("utf-8")), mimetype="text/plain", resumable=False
        )
        self._service.files().update(
            fileId=self._file_ids[doc], media_body=media
        ).execute()
