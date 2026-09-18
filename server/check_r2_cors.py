import boto3

s3 = boto3.client(
    's3',
    endpoint_url='https://f966dfa6f68da7e179742fcf3f565b24.r2.cloudflarestorage.com',
    aws_access_key_id='f1fa6aa7ef86806d8f63bc3ccba7bf32',
    aws_secret_access_key='abea02e882ce66d2e645c1ca2db3b6e0c9efbfe14616aeac5745e7e72bad68e0',
    region_name='auto'
)

try:
    response = s3.get_bucket_cors(Bucket='footballai')
    print("CORS Config:", response.get('CORSRules'))
except Exception as e:
    print("Error:", e)
