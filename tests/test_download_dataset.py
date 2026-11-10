"""Offline tests only: archives contain synthetic schema fixtures."""
import subprocess
import zipfile
from unittest.mock import Mock

import pytest
from scripts import download_dataset as downloader


def archive(path, frame=None, members=None):
    with zipfile.ZipFile(path, 'w') as bundle:
        if frame is not None:
            bundle.writestr('cs-training.csv', frame.to_csv(index=True))
            bundle.writestr('cs-test.csv', 'test download companion file')
        for name, content in (members or {}).items():
            bundle.writestr(name, content)
    return path


def fake_download(monkeypatch, frame):
    def fake(directory):
        return archive(directory / 'GiveMeSomeCredit.zip', frame)
    mock = Mock(side_effect=fake)
    monkeypatch.setattr(downloader, 'download_archive', mock)
    return mock


def test_download_extract_and_skip(frame, tmp_path, monkeypatch):
    mock = fake_download(monkeypatch, frame)
    result = downloader.download_dataset(raw_dir=tmp_path)
    assert result.name == 'cs-training.csv'
    assert len(downloader.load_data(result)) == len(frame)
    assert (tmp_path / 'cs-test.csv').is_file()
    assert not list(tmp_path.glob('.download-*'))
    downloader.download_dataset(raw_dir=tmp_path)
    assert mock.call_count == 1
    downloader.download_dataset(force=True, raw_dir=tmp_path)
    assert mock.call_count == 2


def test_invalid_existing_is_replaced(frame, tmp_path, monkeypatch):
    (tmp_path / 'cs-training.csv').write_text('wrong\n1\n')
    mock = fake_download(monkeypatch, frame)
    downloader.download_dataset(raw_dir=tmp_path)
    mock.assert_called_once()
    assert len(downloader.load_data(tmp_path / 'cs-training.csv')) == len(frame)


@pytest.mark.parametrize('kind', ['corrupt', 'missing', 'schema', 'traversal', 'absolute', 'backslash'])
def test_bad_archives_preserve_existing(frame, tmp_path, monkeypatch, kind):
    destination = tmp_path / 'cs-training.csv'
    frame.to_csv(destination, index=False)
    original = destination.read_bytes()
    def bad(directory):
        path = directory / 'bad.zip'
        if kind == 'corrupt':
            path.write_bytes(b'not a zip')
            return path
        members = {'missing': {'cs-test.csv':'x'}, 'schema': {'cs-training.csv':'wrong\n1'},
                   'traversal': {'../escape.txt':'x'}, 'absolute': {'/escape.txt':'x'},
                   'backslash': {'..\\escape.txt':'x'}}[kind]
        return archive(path, members=members)
    monkeypatch.setattr(downloader, 'download_archive', bad)
    with pytest.raises(downloader.DownloadError):
        downloader.download_dataset(force=True, raw_dir=tmp_path)
    assert destination.read_bytes() == original
    assert not list(tmp_path.glob('.download-*'))
    assert not (tmp_path.parent / 'escape.txt').exists()


@pytest.mark.parametrize(('output','expected'), [
    ('401 Unauthorized secret-token', 'authentication'),
    ('Could not find kaggle.json', 'authentication'),
    ('403 Forbidden secret-token', 'accept its rules'),
    ('ConnectionError secret-token', 'network'),
    ('No module named kaggle', 'requirements.txt'),
])
def test_cli_errors_are_actionable_and_redacted(tmp_path, monkeypatch, output, expected):
    mock = Mock(return_value=subprocess.CompletedProcess([], 1, '', output))
    monkeypatch.setattr(downloader.subprocess, 'run', mock)
    with pytest.raises(downloader.DownloadError) as error:
        downloader.download_archive(tmp_path)
    assert expected in str(error.value)
    assert 'secret-token' not in str(error.value)


def test_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader.subprocess, 'run', Mock(side_effect=subprocess.TimeoutExpired('kaggle', 300)))
    with pytest.raises(downloader.DownloadError, match='timed out'):
        downloader.download_archive(tmp_path)


def test_official_command(tmp_path, monkeypatch):
    archive(tmp_path / 'GiveMeSomeCredit.zip', members={'cs-training.csv':'x'})
    mock = Mock(return_value=subprocess.CompletedProcess([], 0, '', ''))
    monkeypatch.setattr(downloader.subprocess, 'run', mock)
    result = downloader.download_archive(tmp_path)
    assert result.name == 'GiveMeSomeCredit.zip'
    command = mock.call_args.args[0]
    assert command[:5] == [downloader.sys.executable, '-m', 'kaggle', 'competitions', 'download']
    assert command[command.index('-c')+1] == 'GiveMeSomeCredit'


def test_missing_download_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader.subprocess, 'run', Mock(return_value=subprocess.CompletedProcess([], 0, '', '')))
    with pytest.raises(downloader.DownloadError, match='ZIP archive'):
        downloader.download_archive(tmp_path)


def test_zip_checksum_failure(tmp_path):
    path = archive(tmp_path / 'bad-crc.zip', members={'cs-training.csv':'unique payload marker'})
    path.write_bytes(path.read_bytes().replace(b'unique payload marker', b'broken payload marker'))
    with pytest.raises(downloader.DownloadError, match='checksum'):
        downloader.extract_and_validate(path, tmp_path / 'extract')
