"""Train, compare and save Steam recommendation models for ITP4524.

    python game_recommendation.py
    python game_recommendation.py --recommend
    python game_recommendation.py --recommend USER_ID

The default run is offline and uses the supplied Steam data snapshot.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(BASE / '.plot_cache'))
os.environ.setdefault('XDG_CACHE_HOME', str(BASE / '.plot_cache'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, normalize

DATA = BASE / 'data'
MODELS = BASE / 'models'
RESULTS = BASE / 'results'
SEED = 4524
MIN_GAMES = 5
PLAY_THRESHOLD = 1.0
NAMES = {'popularity': 'Popularity', 'content_tags': 'Content: tags',
         'content_full': 'Content: full features', 'user_cf': 'User-based CF'}
FEATURES = ['price_usd', 'positive_percent', 'log_review_count', 'release_year',
            'is_free', 'single_player', 'multi_player', 'co_op', 'windows', 'mac', 'linux']


def prepare_data():
    raw = pd.read_csv(DATA / 'raw' / 'steam_actions.csv', dtype={'user_id': str})
    games = pd.read_csv(DATA / 'raw' / 'steam_games.csv').sort_values('appid').reset_index(drop=True)
    assert games.appid.is_unique, 'Ambiguous app mappings must be resolved first.'
    original_rows = len(raw)
    duplicates = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()
    raw['value'] = pd.to_numeric(raw.value, errors='coerce')
    invalid = raw[['user_id', 'game_name', 'action', 'value']].isna().any(axis=1) | (raw.value < 0)
    missing_rows = int(invalid.sum())
    raw = raw.loc[~invalid]
    played = raw[(raw.action == 'play') & (raw.value >= PLAY_THRESHOLD)].copy()
    played = played.groupby(['user_id', 'game_name'], as_index=False).value.max()
    mapped = played.merge(games[['dataset_game_name', 'appid']], left_on='game_name',
                          right_on='dataset_game_name', validate='many_to_one')
    mapped = mapped.groupby(['user_id', 'appid'], as_index=False).value.max()
    counts = mapped.groupby('user_id').appid.nunique()
    eligible = counts[counts >= MIN_GAMES].index
    interactions = mapped[mapped.user_id.isin(eligible)].rename(columns={'value': 'play_hours'})
    interactions = interactions.sort_values(['user_id', 'appid']).reset_index(drop=True)
    games['log_review_count'] = np.log1p(games.review_count.clip(lower=0))
    games['tag_document'] = games.apply(lambda row: '|'.join(sorted({
        token.strip().casefold() for field in ['tags', 'genres']
        for token in str(row[field]).split('|') if token.strip() and token != 'nan'})), axis=1)
    (DATA / 'processed').mkdir(parents=True, exist_ok=True)
    games.to_csv(DATA / 'processed' / 'games.csv', index=False)
    interactions.to_csv(DATA / 'processed' / 'positive_interactions.csv', index=False)
    missing = games[['price_usd', 'positive_percent', 'review_count', 'release_year',
                     'critic_score', 'tags', 'genres']].isna().sum().rename('missing_count')
    missing.to_csv(RESULTS / 'missing_values.csv', index_label='feature')
    summary = {'original_action_rows': original_rows, 'duplicate_action_rows': duplicates,
               'invalid_action_rows_removed': missing_rows,
               'positive_definition': 'Observed play time >= 1 hour; an implicit interest proxy.',
               'matched_positive_interactions_before_user_filter': len(mapped),
               'matched_users_before_user_filter': int(mapped.user_id.nunique()),
               'minimum_distinct_games_per_user': MIN_GAMES,
               'eligible_users': int(interactions.user_id.nunique()),
               'positive_interactions': len(interactions), 'catalogue_games': len(games),
               'density': len(interactions) / (interactions.user_id.nunique() * len(games))}
    return games, interactions, summary


def make_splits(interactions):
    rng = np.random.default_rng(SEED)
    pieces = {'training': [], 'validation': [], 'testing': []}
    for _, rows in interactions.groupby('user_id', sort=True):
        indices = rng.permutation(len(rows))
        n_test = max(1, int(len(rows) * .2))
        n_val = max(1, int(len(rows) * .1))
        pieces['testing'].append(rows.iloc[indices[:n_test]])
        pieces['validation'].append(rows.iloc[indices[n_test:n_test + n_val]])
        pieces['training'].append(rows.iloc[indices[n_test + n_val:]])
    splits = {name: pd.concat(rows).sort_values(['user_id', 'appid']).reset_index(drop=True)
              for name, rows in pieces.items()}
    sets = [set(zip(frame.user_id, frame.appid)) for frame in splits.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert sum(len(x) for x in splits.values()) == len(interactions)
    assert all(frame.groupby('user_id').size().min() >= 1 for frame in splits.values())
    (DATA / 'splits').mkdir(exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(DATA / 'splits' / f'{name}.csv', index=False)
    return splits


def interaction_matrix(frame, users, appids):
    user_index = {key: i for i, key in enumerate(users)}
    item_index = {key: i for i, key in enumerate(appids)}
    row = frame.user_id.map(user_index).to_numpy()
    col = frame.appid.map(item_index).to_numpy()
    return sparse.csr_matrix((np.ones(len(frame)), (row, col)), shape=(len(users), len(appids)))


def fit_models(training, games, alpha=.75, neighbours=20):
    users = sorted(training.user_id.unique())
    appids = games.appid.astype(int).tolist()
    matrix = interaction_matrix(training, users, appids)
    train_games = games.appid.isin(training.appid.unique())
    # The catalogue is known, but learned IDF, medians and scales use training items only.
    vectorizer = TfidfVectorizer(token_pattern=r'[^|]+', binary=True, norm='l2')
    vectorizer.fit(games.loc[train_games, 'tag_document'])
    tags = vectorizer.transform(games.tag_document)
    imputer = SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)
    imputed_train = imputer.fit_transform(games.loc[train_games, FEATURES])
    scaler = MinMaxScaler(clip=True).fit(imputed_train)
    numeric = normalize(scaler.transform(imputer.transform(games[FEATURES])))
    full = normalize(sparse.hstack([tags * np.sqrt(alpha),
                                    sparse.csr_matrix(numeric) * np.sqrt(1 - alpha)], format='csr'))
    nn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=1).fit(normalize(matrix))
    common = {'users': users, 'appids': appids, 'user_item': matrix,
              'popularity': np.asarray(matrix.sum(axis=0)).ravel(),
              'training_games': games.loc[train_games, 'appid'].astype(int).tolist(),
              'sklearn_version': sklearn.__version__, 'seed': SEED}
    models = {
        'popularity': {**common, 'kind': 'popularity'},
        'content_tags': {**common, 'kind': 'content_tags', 'item_features': tags,
                         'vectorizer': vectorizer},
        'content_full': {**common, 'kind': 'content_full', 'item_features': full,
                         'vectorizer': vectorizer, 'imputer': imputer,
                         'scaler': scaler, 'numeric_features': FEATURES, 'tag_weight': alpha},
        'user_cf': {**common, 'kind': 'user_cf', 'neighbours': neighbours, 'nn': nn},
    }
    return models


def score_model(model, queries):
    if model['kind'] == 'popularity':
        return np.tile(model['popularity'], (queries.shape[0], 1))
    if model['kind'].startswith('content'):
        profiles = normalize(queries @ model['item_features'])
        return (profiles @ model['item_features'].T).toarray()
    # Similarity-weighted positive support, excluding the target user itself.
    count = min(model['neighbours'] + 1, len(model['users']))
    distances, indices = model['nn'].kneighbors(normalize(queries), n_neighbors=count)
    row_ids, col_ids, weights = [], [], []
    for i, (distances_row, indices_row) in enumerate(zip(distances, indices)):
        kept = 0
        for distance, peer in zip(distances_row, indices_row):
            weight = max(0., 1. - float(distance))
            if peer == i or weight <= 0:
                continue
            row_ids.append(i); col_ids.append(peer); weights.append(weight)
            kept += 1
            if kept == model['neighbours']:
                break
    support = sparse.csr_matrix((weights, (row_ids, col_ids)), shape=(len(model['users']), len(model['users'])))
    denominator = np.asarray(support.sum(axis=1)).ravel()
    inverse = np.divide(1., denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return (sparse.diags(inverse) @ support @ model['user_item']).toarray()


def rank_scores(scores, model, cutoff=10):
    assert np.isfinite(scores).all()
    ranks = []
    appids = np.asarray(model['appids'])
    for i, values in enumerate(scores):
        values = np.round(values, 12)
        values[model['user_item'][i].indices] = -np.inf
        order = np.lexsort((appids, -model['popularity'], -values))
        order = order[np.isfinite(values[order])][:cutoff]
        assert len(order) == cutoff and len(set(order)) == cutoff
        ranks.append(order)
    return np.asarray(ranks)


def short_queries(training, model):
    # Choose one observed game per user by play time, never by validation/test hits.
    one = training.sort_values(['user_id', 'play_hours', 'appid'], ascending=[True, False, True]).drop_duplicates('user_id')
    return interaction_matrix(one, model['users'], model['appids'])


def evaluate(model, targets, queries):
    scores = score_model(model, queries)
    rankings = rank_scores(scores, model)
    user_index = {key: i for i, key in enumerate(model['users'])}
    item_index = {key: i for i, key in enumerate(model['appids'])}
    truth = [set() for _ in model['users']]
    for user, appid in zip(targets.user_id, targets.appid):
        truth[user_index[user]].add(item_index[appid])
    rows = []
    for cutoff in (5, 10):
        hits = np.asarray([len(set(ranks[:cutoff]) & relevant) for ranks, relevant in zip(rankings, truth)])
        sizes = np.asarray([len(relevant) for relevant in truth])
        assert (sizes > 0).all()
        rows.append({'k': cutoff, 'precision': float(np.mean(hits / cutoff)),
                     'recall': float(np.mean(hits / sizes)),
                     'hit_rate': float(np.mean(hits > 0)),
                     'coverage': len(set(rankings[:, :cutoff].ravel())) / len(model['appids']),
                     'users': len(truth)})
    return rows, rankings, scores


def choose_parameters(training, validation, games):
    settings = [('popularity', .75, 20), ('content_tags', .75, 20),
                ('content_full', .5, 20), ('content_full', .75, 20),
                ('user_cf', .75, 10), ('user_cf', .75, 30)]
    rows = []
    for method, alpha, neighbours in settings:
        model = fit_models(training, games, alpha, neighbours)[method]
        measures, _, _ = evaluate(model, validation, model['user_item'])
        for result in measures:
            rows.append({'model': method, 'tag_weight': alpha if method == 'content_full' else None,
                         'neighbours': neighbours if method == 'user_cf' else None, **result})
    tuning = pd.DataFrame(rows)
    tuning.to_csv(RESULTS / 'validation_results.csv', index=False)
    at_five = tuning[tuning.k == 5].sort_values(['recall', 'precision', 'model'], ascending=[False, False, True])
    alpha = float(at_five[at_five.model == 'content_full'].iloc[0].tag_weight)
    neighbours = int(at_five[at_five.model == 'user_cf'].iloc[0].neighbours)
    chosen = at_five.drop_duplicates('model').iloc[0].model
    return alpha, neighbours, chosen


def make_plots(games, training, measurements):
    charts = RESULTS / 'plots'; charts.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    wine, stone, green = '#703344', '#ac9481', '#48665a'
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].hist(games.price_usd.dropna(), bins=15, color=wine, edgecolor='white')
    axes[0].set(xlabel='Current US price (USD)', ylabel='Games', title='Price distribution')
    axes[1].hist(games.positive_percent.dropna(), bins=15, color=stone, edgecolor='white')
    axes[1].set(xlabel='Positive Steam reviews (%)', ylabel='Games', title='Approval distribution')
    axes[2].hist(np.log10(games.review_count + 1), bins=15, color=green, edgecolor='white')
    axes[2].set(xlabel='log10(review count + 1)', ylabel='Games', title='Review volume')
    fig.tight_layout();fig.savefig(charts / 'catalogue_distributions.png', dpi=170);plt.close(fig)
    counts = Counter(g for text in games.genres.fillna('') for g in text.split('|') if g)
    top = pd.Series(counts).sort_values().tail(10)
    fig, ax = plt.subplots(figsize=(8, 4));top.plot.barh(ax=ax, color=wine)
    ax.set(xlabel='Games (multiple genres per game)', title='Catalogue genres');fig.tight_layout()
    fig.savefig(charts / 'genre_distribution.png', dpi=170);plt.close(fig)
    valid = games.dropna(subset=['price_usd', 'positive_percent'])
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for free, color, label in [(0, wine, 'Paid'), (1, green, 'Free-to-play')]:
        part = valid[valid.is_free == free]
        ax.scatter(part.price_usd, part.positive_percent, s=12 + np.log1p(part.review_count) * 5,
                   alpha=.55, color=color, edgecolors='none', label=label)
    ax.set(xlabel='Current US price (USD)', ylabel='Positive Steam reviews (%)',
           title='Price and approval: descriptive comparison')
    ax.legend(frameon=False);fig.tight_layout();fig.savefig(charts / 'price_and_approval.png', dpi=170);plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(training.groupby('user_id').size(), bins=20, color=wine, edgecolor='white')
    axes[0].set(xlabel='Observed games per user', ylabel='Users', title='Final training histories')
    popular = training.groupby('appid').user_id.nunique().nlargest(10)
    labels = games.set_index('appid').loc[popular.index, 'title'].str.slice(0, 30)
    axes[1].barh(np.arange(len(popular)), popular.values, color=stone)
    axes[1].set_yticks(np.arange(len(popular)), labels);axes[1].invert_yaxis()
    axes[1].set(xlabel='Training users', title='Most frequent training games')
    fig.tight_layout();fig.savefig(charts / 'interaction_distribution.png', dpi=170);plt.close(fig)
    at_five = measurements[measurements.k == 5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in zip(axes, ['recall', 'precision'], ['Recall@5', 'Precision@5']):
        x = np.arange(len(NAMES));width = .35
        for offset, history, color in [(-width / 2, 'full', wine), (width / 2, 'one_game', stone)]:
            values = at_five[at_five.history == history].set_index('model').loc[list(NAMES), metric]
            ax.bar(x + offset, values, width, color=color, label='Full history' if history == 'full' else 'One-game query')
        ax.set_xticks(x, ['Popularity', 'Content:\ntags', 'Content:\nfull', 'User CF'])
        ax.set(ylabel=title, title=title);ax.legend(frameon=False, fontsize=8)
    fig.tight_layout();fig.savefig(charts / 'test_comparison.png', dpi=170);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(NAMES));width = .35
    for offset, history, color in [(-width/2, 'full', wine), (width/2, 'one_game', stone)]:
        values = at_five[at_five.history == history].set_index('model').loc[list(NAMES), 'recall']
        ax.bar(x+offset, values, width, color=color, label='Full history' if history=='full' else 'One-game query')
    ax.set_xticks(x, ['Popularity', 'Content: tags', 'Content: full', 'User CF'])
    ax.set(ylabel='Recall@5', title='Held-out recall under two query settings')
    ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(charts / 'recall_at5.png', dpi=170);plt.close(fig)


def example_rows(models, training, games, user_id=None):
    first = next(iter(models.values()))
    sizes = training.groupby('user_id').size()
    if user_id is None:
        user_id = sorted(sizes.index, key=lambda key: (abs(sizes[key] - sizes.median()), key))[0]
    if user_id not in first['users']:
        raise ValueError('Unknown user. Run --recommend without an ID to see a saved example.')
    index = first['users'].index(user_id)
    lookup = games.set_index('appid')
    rows = []
    for name, model in models.items():
        scores = score_model(model, model['user_item'])
        ranks = rank_scores(scores, model)[index]
        for rank, item in enumerate(ranks, 1):
            appid = model['appids'][item];game = lookup.loc[appid]
            rows.append({'user_id': user_id, 'model': name, 'rank': rank, 'appid': appid,
                         'title': game.title, 'score': scores[index, item],
                         'price_usd': game.price_usd, 'positive_percent': game.positive_percent,
                         'review_count': game.review_count})
    history = training[training.user_id == user_id].merge(games[['appid', 'title']], on='appid')
    return user_id, pd.DataFrame(rows), history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recommend', nargs='?', const='example', metavar='USER_ID',
                        help='Load saved models and recommend for an existing pseudonymised user.')
    args = parser.parse_args()
    for folder in [MODELS, RESULTS]:folder.mkdir(parents=True, exist_ok=True)
    if args.recommend:
        models = {name: joblib.load(MODELS / f'{name}.joblib') for name in NAMES}
        games = pd.read_csv(DATA / 'processed' / 'games.csv')
        training = pd.read_csv(DATA / 'splits' / 'final_training.csv', dtype={'user_id': str})
        user, rows, history = example_rows(models, training, games,
                                          None if args.recommend == 'example' else args.recommend)
        print('User:', user);print('\nObserved history:\n', history[['title', 'play_hours']].to_string(index=False))
        print('\nSaved-model recommendations:\n', rows[rows['rank'] <= 5][['model', 'rank', 'title', 'score']].to_string(index=False))
        return
    games, interactions, summary = prepare_data()
    splits = make_splits(interactions)
    print(f"Prepared {summary['eligible_users']:,} users, {summary['catalogue_games']} games and {len(interactions):,} positive interactions.", flush=True)
    alpha, neighbours, selected = choose_parameters(splits['training'], splits['validation'], games)
    print(f'Validation selected: {selected}; full-content tag weight={alpha}; CF neighbours={neighbours}.', flush=True)
    final_training = pd.concat([splits['training'], splits['validation']]).sort_values(['user_id', 'appid']).reset_index(drop=True)
    final_training.to_csv(DATA / 'splits' / 'final_training.csv', index=False)
    models = fit_models(final_training, games, alpha, neighbours)
    measurements, per_user = [], []
    for name, model in models.items():
        for history, queries in [('full', model['user_item']), ('one_game', short_queries(final_training, model))]:
            rows, ranks, _ = evaluate(model, splits['testing'], queries)
            for row in rows:measurements.append({'model': name, 'history': history, **row})
            truth = splits['testing'].groupby('user_id').appid.apply(set).to_dict()
            for user, items in zip(model['users'], ranks):
                predicted = [model['appids'][i] for i in items]
                per_user.append({'user_id': user, 'model': name, 'history': history,
                                 'test_items': '|'.join(map(str, sorted(truth[user]))),
                                 'recommended_items': '|'.join(map(str, predicted)),
                                 'hits_at_5': len(set(predicted[:5]) & truth[user])})
        joblib.dump(model, MODELS / f'{name}.joblib', compress=3)
    joblib.dump(models[selected], MODELS / 'selected_model.joblib', compress=3)
    metrics = pd.DataFrame(measurements)
    metrics.to_csv(RESULTS / 'test_metrics.csv', index=False)
    pd.DataFrame(per_user).to_csv(RESULTS / 'per_user_results.csv', index=False)
    user, examples, observed = example_rows(models, final_training, games)
    examples.to_csv(RESULTS / 'example_recommendations.csv', index=False)
    observed.to_csv(RESULTS / 'example_history.csv', index=False)
    summary.update({'split_seed': SEED, 'split_type': 'Per-user random holdout; no timestamps are available.',
                    'split_rows': {key: len(value) for key, value in splits.items()},
                    'final_training_rows': len(final_training), 'selected_by_validation': selected,
                    'content_full_tag_weight': alpha, 'cf_neighbours': neighbours,
                    'example_user_selected_from_training_only': user,
                    'versions': {'python': sys.version.split()[0], 'numpy': np.__version__,
                                 'pandas': pd.__version__, 'sklearn': sklearn.__version__},
                    'generated_at_utc': datetime.now(timezone.utc).isoformat()})
    (RESULTS / 'experiment_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (MODELS / 'model_manifest.json').write_text(json.dumps({
        'selected_model': selected, 'selection_metric': 'Validation Recall@5, then Precision@5',
        'feature_columns': FEATURES, 'versions': summary['versions'],
        'final_fit': 'Training plus validation; testing interactions excluded.',
        'ranking_ties': 'Scores rounded to 12 decimal places, then training popularity and app ID. Zero-support CF uses this popularity fallback.',
    }, indent=2) + '\n')
    make_plots(games, final_training, metrics)
    print('\nHeld-out test results:')
    print(metrics[metrics.k == 5][['model', 'history', 'precision', 'recall', 'hit_rate', 'coverage']].to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nExample user: {user}\nSaved models: {MODELS}\nResults: {RESULTS}')
    print('Run python game_recommendation.py --recommend to demonstrate the saved models.')


if __name__ == '__main__':
    main()
