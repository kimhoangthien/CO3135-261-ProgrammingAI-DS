"""Download official competition data; run from any cwd with the project Python."""
import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data import load_data

COMPETITION = 'GiveMeSomeCredit'
RAW_DIR = ROOT / 'data' / 'raw'
AUTH_HELP = (
    'Configure Kaggle authentication: open https://www.kaggle.com/settings/api, '
    'generate a token and save it in ~/.kaggle/access_token, or create Legacy API '
    'Credentials and save kaggle.json in ~/.kaggle/kaggle.json '
    '(Windows: %USERPROFILE%\\.kaggle\\kaggle.json). '
    'Alternatively set KAGGLE_API_TOKEN or both KAGGLE_USERNAME and KAGGLE_KEY. '
    'Never put credentials in this repository.'
)
RULES_HELP = (
    'Sign in at https://www.kaggle.com/competitions/GiveMeSomeCredit/rules '
    'with the account used by your API credentials, join the competition and '
    'accept its rules, then rerun this script.'
)


class DownloadError(RuntimeError):
    """Actionable download failure, safe to print without server credentials."""


def download_archive(directory):
    """Delegate authentication and transfer to the official Kaggle CLI."""
    command = [sys.executable, '-m', 'kaggle', 'competitions', 'download',
               '-c', COMPETITION, '-p', str(directory), '--force', '--quiet']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        raise DownloadError('Kaggle download timed out after 300 seconds. Check your network and retry.') from None
    except OSError:
        raise DownloadError('Could not launch Kaggle CLI. Run uv sync --locked and retry. requirements.txt is a generated compatibility export.') from None
    if result.returncode:
        # Never echo raw CLI output: it may contain tokens or signed download URLs.
        message = (result.stdout + result.stderr).lower()
        if 'no module named kaggle' in message:
            raise DownloadError('Install the official CLI: uv sync --locked. requirements.txt is a generated compatibility export.')
        if any(word in message for word in ['403', 'forbidden', 'permission', 'rules']):
            raise DownloadError('Kaggle denied competition access. ' + RULES_HELP + ' If already accepted, check token permissions.')
        if any(word in message for word in ['401', 'unauthorized', 'credential', 'authenticate', 'authentication', 'api token', 'access_token', 'kaggle.json', 'login', 'log in']):
            raise DownloadError('Kaggle authentication is missing or invalid. ' + AUTH_HELP)
        raise DownloadError('Kaggle download failed. Check network/proxy access and Kaggle service status; '
                            'verify authentication and competition access. ' + AUTH_HELP + ' ' + RULES_HELP)
    archives = list(Path(directory).glob('*.zip'))
    if len(archives) != 1:
        raise DownloadError('Kaggle did not produce exactly one ZIP archive. Retry the download.')
    return archives[0]


def extract_and_validate(archive, staging):
    """Check archive paths, CRCs and schema before publishing any data."""
    staging = Path(staging)
    try:
        with zipfile.ZipFile(archive) as bundle:
            names = set()
            total = 0
            for member in bundle.infolist():
                name = member.filename
                path = PurePosixPath(name)
                if (not name or '\\' in name or ':' in name or path.is_absolute()
                        or '..' in path.parts or (member.external_attr >> 16) & 0o170000 == 0o120000):
                    raise DownloadError('Unsafe ZIP member path; archive was not extracted.')
                if name in names:
                    raise DownloadError('Duplicate ZIP member names; archive was not extracted.')
                names.add(name)
                total += member.file_size
            if total > 1024**3:
                raise DownloadError('Archive exceeds the 1 GiB extraction limit; verify the official download.')
            if bundle.testzip() is not None:
                raise DownloadError('ZIP checksum failed. Download again with --force.')
            bundle.extractall(staging)
    except (zipfile.BadZipFile, EOFError, RuntimeError, NotImplementedError, zlib.error) as exc:
        if isinstance(exc, DownloadError):
            raise
        raise DownloadError('Corrupted or unsupported ZIP archive. Download again with --force.') from None
    training = staging / 'cs-training.csv'
    if not training.is_file():
        raise DownloadError('Archive does not contain cs-training.csv at its root. Verify competition download.')
    try:
        load_data(training)
    except (ValueError, OSError) as exc:
        raise DownloadError(f'Downloaded cs-training.csv failed schema validation ({type(exc).__name__}). '
                            'No existing dataset was replaced; retry with --force.') from None
    return training


def download_dataset(force=False, raw_dir=RAW_DIR):
    raw_dir = Path(raw_dir).resolve()
    destination = raw_dir / 'cs-training.csv'
    if destination.is_file() and not force:
        try:
            load_data(destination)
        except (ValueError, OSError):
            print('Existing training CSV failed validation; downloading a fresh copy.')
        else:
            print(f'Validated existing dataset; skipping download: {destination}')
            return destination
    raw_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.download-', dir=raw_dir) as temporary:
        work = Path(temporary)
        archive = download_archive(work)
        staging = work / 'extracted'
        extract_and_validate(archive, staging)
        # Publish the required training file last. Failed transfer/extraction/schema
        # validation cannot replace a previously valid training CSV.
        files = [p for p in staging.rglob('*') if p.is_file()]
        files.sort(key=lambda p: p.name == 'cs-training.csv')
        for source in files:
            target = raw_dir / source.relative_to(staging)
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
    if not destination.is_file():
        raise DownloadError('Training CSV missing after extraction.')
    print(f'Downloaded and validated official dataset: {destination}')
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force', action='store_true', help='Download again even if the local CSV is valid')
    args = parser.parse_args()
    try:
        download_dataset(force=args.force)
    except (DownloadError, OSError) as exc:
        if isinstance(exc, DownloadError):
            print(f'Error: {exc}', file=sys.stderr)
        else:
            print('Error: Cannot read/write dataset files. Check disk space and filesystem permissions.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
