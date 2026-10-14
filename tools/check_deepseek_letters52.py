"""Offline check of the official tokenizer archive; optional dev dependency: tokenizers.

Usage: python tools/check_deepseek_letters52.py /path/to/deepseek_v4_tokenizer.zip
No downloads, model requests, credential reads or archive extraction.
"""
import hashlib
import json
import string
import sys
import zipfile
from pathlib import Path
from tokenizers import Tokenizer


def verify(path):
    archive = Path(path).read_bytes()
    with zipfile.ZipFile(path) as z:
        raw = z.read('deepseek_v4_tokenizer/tokenizer.json')
    tokenizer = Tokenizer.from_str(raw.decode())
    ids = {}
    for marker in string.ascii_lowercase + string.ascii_uppercase:
        encoded = tokenizer.encode(marker, add_special_tokens=False)
        if len(encoded.ids) != 1 or tokenizer.decode(encoded.ids) != marker:
            raise ValueError(f'Not one exact token: {marker}')
        ids[marker] = encoded.ids[0]
    if len(set(ids.values())) != 52:
        raise ValueError('Token IDs merged across markers')
    return {'archive_sha256': hashlib.sha256(archive).hexdigest(),
            'tokenizer_json_sha256': hashlib.sha256(raw).hexdigest(), 'token_ids': ids}


if __name__ == '__main__':
    print(json.dumps(verify(sys.argv[1]), indent=2))
