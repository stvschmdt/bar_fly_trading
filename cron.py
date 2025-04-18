import atexit
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from constants import BASE_DIR

DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.file']
LOCK_FILE = Path('/tmp/cron.lock')
TABLE_IMAGE_JPG = BASE_DIR / 'table_image.jpg'
CREDENTIALS_PATH = BASE_DIR / 'service_account_credentials.json'
DRIVE_FOLDER_ID = '1UqjZP_QPqD0tP82cqLhBWmR6B3zDV7fe'


def clean_up():
    """Remove the lock file on exit."""
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()

    # Delete table_image.png, overnight_<date> and screener_results_<date>.csv
    if TABLE_IMAGE_JPG.exists():
        TABLE_IMAGE_JPG.unlink()
    for path in BASE_DIR.glob('overnight_*'):
        if path.suffix == '.pdf' or path.is_dir():
            path.unlink(missing_ok=True) if path.is_file() else shutil.rmtree(path, ignore_errors=True)
    for file in BASE_DIR.glob('screener_results_*.csv'):
        file.unlink()


# Authenticate
def authenticate_drive():
    creds = None
    if creds is None:
        creds = service_account.Credentials.from_service_account_file(str(CREDENTIALS_PATH), scopes=DRIVE_SCOPES)
    return creds


# Upload File
def upload_to_drive(file_path, file_name):
    creds = authenticate_drive()
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': file_name, 'parents': [DRIVE_FOLDER_ID]}
    media = MediaFileUpload(file_path, mimetype='application/pdf', chunksize=20*1024*1024, resumable=True)
    print(f"Starting upload to drive at {str(datetime.now())}")
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"Uploaded File ID: {file.get('id')} at {str(datetime.now())}")


def main():
    # Check if the lock file exists
    if LOCK_FILE.exists():
        print('Script already running')
        sys.exit(1)

    # Create the lock file
    LOCK_FILE.touch()

    # Register cleanup function
    atexit.register(clean_up)

    # Register cleanup for SIGINT and SIGTERM
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

    os.chdir(BASE_DIR)

    # Pull API data, run screener and create overnight PDF
    subprocess.run(
        [sys.executable, '-m', 'api_data.pull_api_data', '-w', 'all'],
        check=True
    )
    subprocess.run(
        [sys.executable, '-m', 'visualizations.screener', '--n_days', '60', '--data', BASE_DIR],
        check=True
    )

    # Upload most recent overnight PDF to Google Drive
    pdf_files = list(BASE_DIR.glob('overnight_*.pdf'))
    if not pdf_files:
        print('found no overnight pdf files')

    most_recent_file = max(pdf_files, key=os.path.getmtime)
    upload_to_drive(most_recent_file.name, most_recent_file.name)


if __name__ == '__main__':
    main()
