from __future__ import absolute_import, division, print_function, unicode_literals

import logging


def extract_weight_order(filename, default_weight_label=None):
    with open(filename) as file:
        for line in file:
            terms = line.replace('"', '').split()

            if terms[0] != 'N':
                continue

            logging.debug('HEPMC line:\n%s', line)
            logging.debug('HEPMC terms: %s', terms)

            n_benchmarks = int(terms[1])
            assert len(terms) == n_benchmarks + 2

            weight_labels = []
            for term in terms[2:]:

                if term.startswith('id='):
                    term = term[3:]
                    term = term.partition('_MERGING=')[0]
                    weight_labels.append(term)
                else:
                    weight_labels.append(default_weight_label)

            logging.debug('HEPMC weights: %s', weight_labels)

            return weight_labels

    raise RuntimeError('HEPMC weight labels not found in file {}'.format(filename))

