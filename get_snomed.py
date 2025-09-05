from dotenv import load_dotenv
import requests
import os

# TODO: move functions to a lib file

import re
def get_snomed_version_number(snomed_file_str: str):
    pattern = re.compile(r'(\d{8})(?=T)')
    matched = pattern.search(file_url)
    if (matched):
        return matched.group(1)
    # else:
    return False

from tqdm import tqdm
# # https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# get API key
load_dotenv()
NHS_KEY = os.getenv('NHS_API_KEY')

# call NHS API to get latest SNOMED release file name + file url
response = requests.get(f"https://isd.digital.nhs.uk/trud/api/v1/keys/{NHS_KEY}/items/4/releases?latest")
JSON_RESPONSE = response.json()

file_url = JSON_RESPONSE['releases'][0]['archiveFileUrl']
file_name = JSON_RESPONSE['releases'][0]['id']
version = get_snomed_version_number(file_name)

print("SNOMED CT File URL: ", file_url)
print("SNOMED CT File Name: ", file_name)
print("SNOMED CT Version:", version)

# download SNOMED CT latest release
download(file_url, f"./data/{file_name}")

# output the verson for re-use via bash script
print(version)