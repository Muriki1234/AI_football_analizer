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
        elif local_path.suffix == ".json":
            content_type = "application/json"
            
        client.upload_file(
            str(local_path), 
            settings.R2_BUCKET_NAME, 
            remote_key,
            ExtraArgs={'ContentType': content_type}
        )
        
        # Construct public URL assuming the bucket is configured for public access
        # For R2, this often requires configuring a custom domain or r2.dev subdomain
        # For MVP, we might just return the presigned URL
        presigned_url = client.generate_presigned_url(
            'get_object',
            Params={'Bucket': settings.R2_BUCKET_NAME, 'Key': remote_key},
            ExpiresIn=3600 * 24 * 7 # 7 days
        )
        return presigned_url
    except ClientError as e:
        log.error(f"R2 upload failed: {e}")
        return None
