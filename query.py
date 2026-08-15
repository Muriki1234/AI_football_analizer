import requests
import json
url = "https://dexhzxtnsyebupowbbmt.supabase.co/rest/v1/sessions?select=id,extra,video_url,created_at&order=created_at.desc&limit=1"
headers = {
    "apikey": "sb_publishable_MIwUU3cApcGqRwWXSzZBwQ_3fj2rPE-",
    "Authorization": "Bearer sb_publishable_MIwUU3cApcGqRwWXSzZBwQ_3fj2rPE-"
}
res = requests.get(url, headers=headers)
print(json.dumps(res.json(), indent=2))
