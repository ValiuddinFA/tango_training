import os
from from_root import from_root

PROJECT_DIRECTORY = 'tango_training'
BASE_DIR = from_root()

LOCAL_FILE_PATH = os.path.join('resources', 'compressed')
FIRST_IMPRESSION_PATH = os.path.join(BASE_DIR,'resources/first-impressions')
BASE_URL = 'http://158.109.8.102/FirstImpressionsV2/'
USER_NAME = os.environ.get('USER_NAME')
PASSWORD = os.environ.get('PASSWORD')
ENCRYPTION_KEY = os.environ.get('encryption_key')
ALT_ENCRYPTION_KEY = os.environ.get('alt_encryption_key')
DOWNLOAD_LINKS = [
        'train-1.zip',
        'train-2.zip',
        'train-3.zip',
        'train-4.zip',
        'train-5.zip',
        'train-6.zip',
        'test-1e.zip',
        'test-2e.zip',
        'val-1.zip',
        'val-2.zip',
    ]
META_DOWNLOADS = [
        'test-annotation-e.zip',
        'test-transcription-e.zip',
        'train-annotation.zip',
        'train-transcription.zip',
        'val-annotation-e.zip',
        'val-transcription.zip'
    ]