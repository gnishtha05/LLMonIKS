import json
import re
from tqdm.auto import tqdm

hindi = 'hindi_bhashya.json'
with open(hindi, "r", encoding="utf-8") as f:
    hindi = json.load(f) 

skt = 'skt_bhashya.json'
with open(skt, "r", encoding="utf-8") as f:
    skt = json.load(f) 

def process_skt_shloka(skt_shloka, skt_bhashya):
    pattern = r'।।\d+(?:\.\d+)?।।'
    skt_shloka = re.sub(pattern + r'\s*$', '', skt_shloka)
    skt_bhashya = re.sub(r'^\s*' + pattern, '', skt_bhashya)
    return skt_shloka.strip(), skt_bhashya.strip()

final_data = {}
unmatched = []
for shloka_id, (hindi_shloka,hindi_bhashya) in tqdm(hindi.items()):
    skt_shloka, skt_bhashya = process_skt_shloka(*skt[shloka_id])
    if skt_shloka != hindi_shloka:
        print(f"Shloka {shloka_id} does not match")
        print(f"Hindi bhashya: {hindi_bhashya}")
        print(f"Sanskrit bhashya: {skt_bhashya}")
        print("--------------------------------")
        unmatched.append(shloka_id)
    else:
        final_data[shloka_id] = {'shloka': skt_shloka, 
                                'bhashya_skt': skt_bhashya, 
                                'bhashya_hindi': hindi_bhashya}
    # break

with open('final_data.json', 'w', encoding='utf-8') as f:
    json.dump(final_data, f, ensure_ascii=False, indent=2)

print('--------------------------------')
print(f"Total unmatched: {len(unmatched)}")
print(f"Unmatched: {unmatched}")
print('--------------------------------')