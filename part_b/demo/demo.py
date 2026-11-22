"""Game recommendation demo based on Lecture 9; Python standard library only.

Run: python demo.py
The browser opens automatically. Close the terminal or press Ctrl+C to stop.
All player ratings are fictional demonstration data, not Steam observations.
"""

import argparse
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import webbrowser

BASE = Path(__file__).resolve().parent
DATA = json.loads((BASE / 'demo_data.json').read_text(encoding='utf-8'))
GAMES = {game['id']: game for game in DATA['games']}
PROFILES = {profile['id']: profile for profile in DATA['profiles']}
METHODS = {'content', 'collaborative', 'popular'}


def cosine(first, second):
    """Cosine similarity between two nonnegative feature dictionaries."""
    numerator = sum(value * second.get(key, 0) for key, value in first.items())
    norm = math.sqrt(sum(v * v for v in first.values()) *
                     sum(v * v for v in second.values()))
    return numerator / norm if norm else 0.0


def recommend(profile_id='lin', method='content', genre='all', limit=6, liked=None):
    """Return genuine calculations on the small, explicitly fictional dataset.

    Content filtering compares the user's liked-game tag profile with each game.
    User-based CF uses cosine similarity on binary liked-game vectors, then
    scores candidates by the weighted support of other example users.
    Popularity counts example users who gave a game a rating of at least 4.
    Scores from different methods have different meanings, not probabilities.
    """
    if method not in METHODS:
        raise ValueError('Unknown recommendation method.')
    if genre != 'all' and genre not in {g['genre'] for g in GAMES.values()}:
        raise ValueError('Unknown genre.')
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 12:
        raise ValueError('Choose between 1 and 12 results.')
    if profile_id == 'new':
        if not isinstance(liked, list):
            raise ValueError('Please provide a list of favourite games.')
        if any(not isinstance(game, str) or game not in GAMES for game in liked):
            raise ValueError('Unknown game.')
        liked = list(dict.fromkeys(liked))
        if len(liked) > 8:
            raise ValueError('Please choose no more than 8 games.')
        ratings = {game: 5 for game in liked}
        player_name = 'New player'
    else:
        if profile_id not in PROFILES:
            raise ValueError('Unknown example player.')
        profile = PROFILES[profile_id]
        ratings = profile['ratings']
        player_name = profile['name']

    liked_ids = [game for game, rating in ratings.items() if rating >= 4]
    user_tags = Counter(tag for game in liked_ids for tag in GAMES[game]['tags'])
    user_vector = {game: 1 for game in liked_ids}
    neighbours = []
    for peer in DATA['profiles']:
        if peer['id'] == profile_id:
            continue
        peer_vector = {game: 1 for game, rating in peer['ratings'].items() if rating >= 4}
        similarity = cosine(user_vector, peer_vector)
        if similarity > 0:
            neighbours.append((similarity, peer, peer_vector))
    total_similarity = sum(item[0] for item in neighbours)
    candidates = []
    eligible_count = 0
    for game_id, game in GAMES.items():
        # All recorded interactions, including low ratings, are excluded.
        if game_id in ratings or (genre != 'all' and game['genre'] != genre):
            continue
        eligible_count += 1
        shared_tags = sorted(set(game['tags']) & set(user_tags),
                             key=lambda tag: (-user_tags[tag], tag))
        if method == 'content':
            score = cosine(user_tags, {tag: 1 for tag in game['tags']})
            if score <= 0:
                continue
            seeds = sorted([key for key in liked_ids
                            if set(GAMES[key]['tags']) & set(game['tags'])], key=lambda key: (
                -len(set(GAMES[key]['tags']) & set(game['tags'])), key))[:2]
            reason = 'Shares tags with ' + ' and '.join(GAMES[key]['title'] for key in seeds) + ', which you liked.'
            detail = 'Shared interests: ' + ', '.join(shared_tags[:3]) + '.'
        elif method == 'collaborative':
            supporters = [(sim, peer) for sim, peer, vector in neighbours if game_id in vector]
            score = sum(sim for sim, _ in supporters) / total_similarity if total_similarity else 0
            if score <= 0:
                continue
            supporters.sort(key=lambda pair: (-pair[0], pair[1]['id']))
            reason = 'Liked by ' + ' and '.join(peer['name'] for _, peer in supporters[:2]) + ', who share some of your favourites.'
            detail = 'Find similar players from shared favourites, then combine their weighted support for this game.'
        else:
            count = sum(peer['ratings'].get(game_id, 0) >= 4 for peer in DATA['profiles'])
            score = float(count)
            if not count:
                continue
            player_label = 'player' if count == 1 else 'players'
            reason = f'{count} example {player_label} rated this game 4 or above.'
            detail = 'Popularity baseline: ranked by example likes, without using your interests. Previously rated games are still excluded.'
        candidates.append({**game, 'score': round(score, 6), 'reason': reason,
                           'detail': detail, 'shared_tags': shared_tags})
    candidates.sort(key=lambda row: (-row['score'], row['title']))
    return {
        'profile_id': profile_id, 'player_name': player_name, 'method': method,
        'results': candidates[:limit], 'eligible_count': eligible_count,
        'matched_count': len(candidates), 'history_count': len(ratings),
        'neighbour_count': len(neighbours),
        'taste_tags': [tag for tag, _ in user_tags.most_common(5)],
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_content(self, status, content, content_type):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(content)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, status, payload):
        self.send_content(status, json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                          'application/json; charset=utf-8')

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self.send_content(200, (BASE / 'index.html').read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/catalog':
            self.send_json(200, DATA)
        elif self.path == '/favicon.ico':
            self.send_content(204, b'', 'image/x-icon')
        else:
            self.send_json(404, {'error': 'Not found.'})

    def do_POST(self):
        if self.path != '/api/recommend':
            self.send_json(404, {'error': 'Not found.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 16384:
                raise ValueError('Invalid request length.')
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('Invalid request.')
            response = recommend(
                profile_id=payload.get('profile_id', 'lin'),
                method=payload.get('method', 'content'),
                genre=payload.get('genre', 'all'),
                limit=payload.get('limit', 6), liked=payload.get('liked'),
            )
        except (ValueError, TypeError) as error:
            self.send_json(400, {'error': str(error)})
            return
        self.send_json(200, response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=0, help='Local port; 0 selects a free port.')
    parser.add_argument('--no-browser', action='store_true', help='Run without opening a browser.')
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    url = f'http://127.0.0.1:{server.server_port}'
    print('\nGame recommendation demo', flush=True)
    print('Open: ' + url, flush=True)
    print('Demo data only. No third-party packages required.', flush=True)
    print('Press Ctrl+C or close this terminal to stop.\n', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nDemo stopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
