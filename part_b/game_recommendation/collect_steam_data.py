"""Collect the Steam catalogue and the public Steam-200k interaction snapshot.

Run only to collect source data. Existing API responses are reused to resume a
collection; remove data/raw/steam_api first to fetch fresh official responses.
The modelling program uses the bundled snapshot without network requests.
Player identifiers are pseudonymised; review text and account names are not used.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / 'data' / 'raw'
CACHE = RAW / 'steam_api'
SOURCE_URL = 'https://raw.githubusercontent.com/guilherminog/raw-steam200k/main/steam-200k.csv'
SOURCE_PAGE = 'https://www.kaggle.com/datasets/tamber/steam-video-games'
TAG_URL = 'https://api.steampowered.com/IStoreService/GetTagList/v1/?language=english'
USER_AGENT = 'ITP4524-GameLens-Course-Research/1.0'
# These are the original app identities, not newer editions with similar names.
ORIGINAL_APP_IDS = {
    'Counter-Strike Global Offensive': (730, 'Original app ID now named Counter-Strike 2'),
    'Grand Theft Auto V': (271590, 'Original app ID now named Grand Theft Auto V Legacy'),
    'The Elder Scrolls V Skyrim': (72850, 'Original Skyrim app, not Special Edition'),
}


def normalise_name(value):
    value = unicodedata.normalize('NFKC', value).casefold()
    value = value.replace('®', '').replace('™', '')
    return re.sub(r'[^a-z0-9]+', '', value)


def read_url(url):
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': USER_AGENT}), timeout=30) as response:
                body = response.read(32 * 1024 * 1024)
                if 'store.steampowered.com' in url and response.geturl() == 'https://store.steampowered.com/':
                    raise ValueError('Steam redirected the data request to its homepage.')
                return body
        except HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(min(30, float(error.headers.get('Retry-After', 3 * (attempt + 1)))))
        except (TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def cached_json(path, url):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    data = json.loads(read_url(url))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    time.sleep(.12)
    return data


def resolve_app(name):
    if name in ORIGINAL_APP_IDS:
        return (*ORIGINAL_APP_IDS[name], 'verified_original_app_id')
    url = 'https://store.steampowered.com/api/storesearch/?' + urlencode(
        {'term': name, 'l': 'english', 'cc': 'us'})
    key = hashlib.sha256(name.encode()).hexdigest()[:16]
    result = cached_json(CACHE / 'search' / (key + '.json'), url)
    matches = [item for item in result.get('items', [])
               if normalise_name(item['name']) == normalise_name(name)]
    if len(matches) != 1:
        raise ValueError('No unique exact title match; edition guesses were not used.')
    return int(matches[0]['id']), 'Exact title after punctuation/trademark normalisation', 'store_search'


def collect_game(name, tag_names):
    app_id, note, match_method = resolve_app(name)
    meta_url = 'https://store.steampowered.com/api/appdetails?' + urlencode(
        {'appids': app_id, 'l': 'english', 'cc': 'us'})
    payload = cached_json(CACHE / 'details' / f'{app_id}.json', meta_url)
    item = payload.get(str(app_id), {})
    if not item.get('success'):
        raise ValueError('Official app details unavailable.')
    game = item['data']
    if game.get('type') != 'game' or int(game['steam_appid']) != app_id:
        raise ValueError('The matched app is not a base game.')
    browse_input = {'ids': [{'appid': app_id}],
                    'context': {'language': 'english', 'country_code': 'US'},
                    'data_request': {'include_reviews': True, 'include_tag_count': 20,
                                     'include_basic_info': True, 'include_release': True,
                                     'include_platforms': True, 'include_categories': True}}
    browse_url = 'https://api.steampowered.com/IStoreBrowseService/GetItems/v1/?' + urlencode(
        {'input_json': json.dumps(browse_input, separators=(',', ':'))})
    browse = cached_json(CACHE / 'browse' / f'{app_id}.json', browse_url)
    items = browse.get('response', {}).get('store_items', [])
    current = next((x for x in items if x.get('appid') == app_id), {})
    summary = current.get('reviews', {}).get('summary_filtered', {})
    tags = [tag_names.get(int(x['tagid']), f"Steam tag {x['tagid']}")
            for x in current.get('tags', [])[:20]]
    genres = [x['description'] for x in game.get('genres', [])]
    categories = [x['description'] for x in game.get('categories', [])]
    price = game.get('price_overview', {})
    free = bool(game.get('is_free', False))
    currency = price.get('currency', 'USD' if free else '')
    usd = currency == 'USD'
    price_usd = 0.0 if free else (price.get('final', 0) / 100 if usd and 'final' in price else None)
    released = current.get('release', {}).get('steam_release_date')
    date_text = game.get('release_date', {}).get('date', '')
    if not released:
        parsed = pd.to_datetime(date_text, errors='coerce')
        released = None if pd.isna(parsed) else int(parsed.timestamp())
    year = datetime.fromtimestamp(released, timezone.utc).year if released else None
    row = {
        'dataset_game_name': name, 'appid': app_id, 'title': game['name'],
        'price_usd': price_usd, 'is_free': int(free),
        'positive_percent': summary.get('percent_positive'),
        'review_count': summary.get('review_count'),
        'release_year': year,
        'windows': int(game.get('platforms', {}).get('windows', False)),
        'mac': int(game.get('platforms', {}).get('mac', False)),
        'linux': int(game.get('platforms', {}).get('linux', False)),
        'genres': '|'.join(genres), 'tags': '|'.join(tags or genres),
        'single_player': int('Single-player' in categories),
        'multi_player': int(any(x in categories for x in ['Multi-player', 'PvP', 'Online PvP'])),
        'co_op': int(any('Co-op' in x for x in categories)),
        'mapping_method': match_method, 'mapping_note': note,
        'metadata_url': meta_url, 'review_summary_url': browse_url,
        'metadata_retrieved_at_utc': datetime.fromtimestamp(
            (CACHE / 'details' / f'{app_id}.json').stat().st_mtime, timezone.utc).isoformat(),
        'review_summary_retrieved_at_utc': datetime.fromtimestamp(
            (CACHE / 'browse' / f'{app_id}.json').stat().st_mtime, timezone.utc).isoformat(),
    }
    return row


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    snapshot = datetime.now(timezone.utc).isoformat()
    original = read_url(SOURCE_URL)
    source_hash = hashlib.sha256(original).hexdigest()
    from io import BytesIO
    actions = pd.read_csv(BytesIO(original), header=None,
                          names=['original_user', 'game_name', 'action', 'value', 'unused'],
                          dtype={'original_user': str})
    original_users = actions.original_user.nunique()
    actions['user_id'] = actions.original_user.map(
        lambda value: hashlib.sha256(('GameLens-Steam200k:' + value).encode()).hexdigest()[:16])
    assert actions.user_id.nunique() == original_users
    actions[['user_id', 'game_name', 'action', 'value']].to_csv(
        RAW / 'steam_actions.csv', index=False)
    play = actions[(actions.action == 'play') & (actions.value >= 1)].drop_duplicates(['user_id', 'game_name'])
    # Fixed catalogue sample; model popularity is later computed from training data only.
    names = play.game_name.value_counts().head(200).index.tolist()
    tag_payload = cached_json(CACHE / 'tag_dictionary.json', TAG_URL)
    tag_names = {int(x['tagid']): x['name'] for x in tag_payload['response']['tags']}
    rows, skipped = [], []
    print(f'Collecting up to {len(names)} matched Steam games; {len(actions):,} original actions.', flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(collect_game, name, tag_names): name for name in names}
        for future in as_completed(futures):
            name = futures[future]
            try:
                row = future.result();rows.append(row)
                print(f"{len(rows):3}: {row['title']} | price={row['price_usd']} | reviews={row['review_count']}", flush=True)
            except Exception as error:
                skipped.append({'dataset_game_name': name, 'reason': str(error)})
                print('Skipped:', name, str(error), flush=True)
    catalogue = pd.DataFrame(rows).sort_values('appid')
    catalogue.to_csv(RAW / 'steam_games.csv', index=False)
    pd.DataFrame(skipped, columns=['dataset_game_name', 'reason']).to_csv(RAW / 'unmatched_games.csv', index=False)
    manifest = {
        'retrieved_at_utc': snapshot, 'catalogue_country': 'US', 'catalogue_language': 'English',
        'interaction_dataset': 'Steam Video Games / Steam-200k', 'canonical_source': SOURCE_PAGE,
        'download_url': SOURCE_URL, 'download_sha256': source_hash,
        'original_rows': len(actions), 'original_users': original_users,
        'original_games': int(actions.game_name.nunique()),
        'identifier_transformation': 'SHA256(GameLens-Steam200k:original_id), first 16 hex characters',
        'sampling_rule': 'Top 200 titles by distinct players with at least one hour in the interaction snapshot; exact title matching plus three documented original-app aliases.',
        'matched_titles': len(catalogue), 'unique_appids': int(catalogue.appid.nunique()),
        'skipped_titles': len(skipped),
        'metadata_cache_policy': 'Existing API responses are reused. Per-response retrieval times are recorded in steam_games.csv; this manifest records the snapshot assembly start.',
        'direct_review_endpoint': 'https://store.steampowered.com/appreviews/620?json=1',
        'direct_review_status': 'Returned HTTP 302 to the storefront homepage in this environment. No individual review data was fabricated.',
        'limitations': [
            'The interaction snapshot is historical and has no event timestamps.',
            'The catalogue contains current metadata, not historical prices at play time.',
            'Play time is implicit feedback, not a verified positive rating.',
            'The sample favours games with many players and users with multiple observed games.',
            'Public source access does not establish an open licence; retain source attribution.',
        ],
    }
    (RAW / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'DONE: {len(catalogue)} matched titles, {catalogue.appid.nunique()} app IDs; {len(skipped)} skipped.', flush=True)


if __name__ == '__main__':
    main()
