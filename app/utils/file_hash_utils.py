import hashlib


def build_file_sha256(local_file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(local_file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def build_doc_id(local_file_path: str) -> str:
    """
    Build a stable document id from file content.

    Same bytes produce the same doc_id, no matter what the uploaded file name is.
    """
    return f"doc_{build_file_sha256(local_file_path)[:16]}"
