"""Utility functions for Google Drive access using user credentials (refresh token)."""
import os
import json

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload


def get_credentials():
    """Load user credentials from GCP_CREDENTIALS env variable (authorized_user format)."""
    creds_info = json.loads(os.environ["GCP_CREDENTIALS"])

    creds = Credentials(
        token=None,
        refresh_token=creds_info["refresh_token"],
        token_uri=creds_info.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=creds_info["client_id"],
        client_secret=creds_info["client_secret"],
    )
    # Refresh to get a valid access token
    creds.refresh(Request())
    return creds


def get_drive_service():
    return build("drive", "v3", credentials=get_credentials())


def upload_file_to_drive(local_path, folder_id, drive_service=None):
    """Upload (or overwrite) a local file to a specific Drive folder."""
    if drive_service is None:
        drive_service = get_drive_service()

    file_name = os.path.basename(local_path)
    existing = drive_service.files().list(
        q=f"name='{file_name}' and '{folder_id}' in parents and trashed=false",
        fields="files(id)",
    ).execute().get("files", [])

    media = MediaFileUpload(local_path, resumable=True)

    if existing:
        drive_service.files().update(fileId=existing[0]["id"], media_body=media).execute()
        print(f"[drive] Updated:  {file_name}")
    else:
        drive_service.files().create(
            body={"name": file_name, "parents": [folder_id]},
            media_body=media,
        ).execute()
        print(f"[drive] Uploaded: {file_name}")


def download_file_from_drive(file_name, folder_id, local_path, drive_service=None):
    """Download a file from a Drive folder to a local path."""
    if drive_service is None:
        drive_service = get_drive_service()

    results = drive_service.files().list(
        q=f"name='{file_name}' and '{folder_id}' in parents and trashed=false",
        fields="files(id)",
    ).execute().get("files", [])

    if not results:
        raise FileNotFoundError(f"File '{file_name}' not found in Drive folder {folder_id}.")

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    request = drive_service.files().get_media(fileId=results[0]["id"])
    with open(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    print(f"[drive] Downloaded: {file_name} → {local_path}")
