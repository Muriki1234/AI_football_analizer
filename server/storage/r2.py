import os
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from ..config import settings
import logging

log = logging.getLogger(__name__)

_r2_client = None

def get_r2_client():
    global _r2_client
    if _r2_client is None and settings.R2_ACCOUNT_ID and settings.R2_ACCESS_KEY_ID:
        try:
            _r2_client = boto3.client(
                's3',
                endpoint_url=settings.R2_ENDPOINT_URL,
                aws_access_key_id=settings.R2_ACCESS_KEY_ID,
                aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
                region_name='auto'
            )
        except Exception as e:
            log.error(f"Failed to initialize R2 client: {e}")
    return _r2_client

def upload_to_r2(local_path: Path, remote_key: str) -> str | None:
    """
    Uploads a file to Cloudflare R2 and returns the public URL.
    Returns None if R2 is not configured or upload fails.
    
    If R2_PUBLIC_URL is configured (e.g. https://pub-xxx.r2.dev), returns a
    permanent public URL.  Otherwise falls back to a 7-day presigned URL.
    """
    client = get_r2_client()
    if not client or not settings.R2_BUCKET_NAME:
        return None
        
    try:
        content_type = "application/octet-stream"
        if local_path.suffix == ".mp4":
            content_type = "video/mp4"
        elif local_path.suffix == ".png":
            content_type = "image/png"
        elif local_path.suffix == ".jpg" or local_path.suffix == ".jpeg":
            content_type = "image/jpeg"
        elif local_path.suffix == ".json":
            content_type = "application/json"
        elif local_path.suffix == ".pkl":
            content_type = "application/octet-stream"
            
        client.upload_file(
            str(local_path), 
            settings.R2_BUCKET_NAME, 
            remote_key,
            ExtraArgs={'ContentType': content_type}
        )
        
        # Prefer permanent public URL (zero egress cost, never expires)
        if settings.R2_PUBLIC_URL:
            public_base = settings.R2_PUBLIC_URL.rstrip('/')
            return f"{public_base}/{remote_key}"
        
        # Fallback: presigned URL (7 days, costs nothing on R2 but will expire)
        presigned_url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.R2_BUCKET_NAME, 'Key': remote_key},
            ExpiresIn=3600 * 24 * 7  # 7 days
        )
        return presigned_url
    except ClientError as e:
        log.error(f"R2 upload failed: {e}")
        return None


def download_from_r2(remote_key: str, local_path: Path) -> bool:
    """
    Downloads a file from R2 to local disk.
    Used by feature tasks to retrieve tracks.pkl when running on a new worker.
    Returns True on success, False on failure.
    """
    client = get_r2_client()
    if not client or not settings.R2_BUCKET_NAME:
        return False

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(settings.R2_BUCKET_NAME, remote_key, str(local_path))
        return True
    except ClientError as e:
        log.error(f"R2 download failed for {remote_key}: {e}")
        return False
