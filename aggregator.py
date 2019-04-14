import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--files', type=str, nargs='+')

args = parser.parse_args()
files = args.files

data = {}

for period in [str(e) for e in range(1800, 2020, 10)]:
    data[period] = {
        'aggregated': {
            'pos': {},
            'chars': {},
            'keywords': [],
            'sentiment': 0,
            'vocab': 0,
            'wps': 0,
            'wl': 0
        },
        'samples': []
    }

for file in files:
    new_json = json.load(open(file))
    data[str(new_json['period'])]['samples'].append(new_json)

for period in data.keys():
    sample_count = len(data[period]['samples'])

    data[period]['aggregated']['pos'] = Counter(
        data[period]['aggregated']['pos'])
    for i in range(sample_count):
        data[period]['aggregated']['pos'] += Counter(
            data[period]['samples'][i]['pos'])
    for key in data[period]['aggregated']['pos'].keys():
        data[period]['aggregated']['pos'][key] /= sample_count
    data[period]['aggregated']['pos'] = dict(data[period]['aggregated']['pos'])

    data[period]['aggregated']['chars'] = Counter(
        data[period]['aggregated']['chars'])
    for i in range(sample_count):
        data[period]['aggregated']['chars'] += Counter(
            data[period]['samples'][i]['chars'])
    for key in data[period]['aggregated']['chars'].keys():
        data[period]['aggregated']['chars'][key] /= sample_count
    data[period]['aggregated']['chars'] = dict(
        data[period]['aggregated']['chars'])

    for i in range(sample_count):
        data[period]['aggregated']['keywords'] += data[period]['samples'][i]['keywords']
    if len(data[period]['aggregated']['keywords']) > 0:
        np.random.shuffle(data[period]['aggregated']['keywords'])
        data[period]['aggregated']['keywords'] = list(
            data[period]['aggregated']['keywords'])[:10]

    for i in range(sample_count):
        data[period]['aggregated']['sentiment'] += data[period]['samples'][i]['sentiment']
    if sample_count > 0:
        data[period]['aggregated']['sentiment'] /= sample_count

    for i in range(sample_count):
        data[period]['aggregated']['vocab'] += data[period]['samples'][i]['vocab']
    if sample_count > 0:
        data[period]['aggregated']['vocab'] /= sample_count

    for i in range(sample_count):
        data[period]['aggregated']['wps'] += data[period]['samples'][i]['wps']
    if sample_count > 0:
        data[period]['aggregated']['wps'] /= sample_count

    for i in range(sample_count):
        data[period]['aggregated']['wl'] += data[period]['samples'][i]['wl']
    if sample_count > 0:
        data[period]['aggregated']['wl'] /= sample_count

for period in [str(e) for e in range(1800, 2020, 10)]:
    data[period].pop('samples')

print(data)
