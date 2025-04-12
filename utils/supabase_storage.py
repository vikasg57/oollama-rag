import os
from supabase import create_client, Client


class SupabaseStorage:
    def __init__(self):
        self.url: str = os.environ.get("SUPABASE_URL")
        self.key: str = os.environ.get("SUPABASE_KEY")
        self.client: Client = create_client(self.url, self.key)
        self.bucket = "context-docs"

    def upload_file(self, file_path: str, upload_path: str) -> str:
        with open(file_path, "rb") as f:
            response = self.client.storage.from_(self.bucket).upload(
                path=upload_path,
                file=f,
                file_options={"cache-control": "3600", "upsert": "true"}
            )
        if response.path:
            return f"{self.bucket}/{response.path}"
        raise Exception("Failed to upload file to Supabase Storage")

    def download_file(self, storage_path: str, local_path: str):
        try:
            file_bytes = self.client.storage.from_(self.bucket).download(storage_path)
            os.makedirs(os.path.dirname(local_path), exist_ok = True)
            with open(local_path, "wb") as f:
                f.write(file_bytes)
        except Exception as e:
            raise RuntimeError(f"Failed to download {storage_path}: {str(e)}")
