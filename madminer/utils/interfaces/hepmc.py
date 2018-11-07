from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import os
from io import open

from madminer.utils.various import call_command


def extract_weight_order(filename, default_weight_label=None):
    # Untar event file
    new_filename, extension = os.path.splitext(filename)
    if extension == ".gz":
        if not os.path.exists(new_filename):
            call_command("gunzip -k {}".format(filename))
        filename = new_filename

    with open(filename, encoding="latin-1") as file:
        for line in file:
            terms = line.replace('"', "").split()

            if len(terms) == 0 or terms[0] != "N":
                continue

            n_benchmarks = int(terms[1])
            assert len(terms) == n_benchmarks + 2

            weight_labels = []
            for term in terms[2:]:

                if term.startswith("id="):
                    term = term[3:]
                    term = term.partition("_MERGING=")[0]
                    weight_labels.append(term)
                else:
                    weight_labels.append(default_weight_label)

            logging.debug("Found weight labels in HEPMC file: %s", weight_labels)

            return weight_labels

    # Default result (no reweighting, background scenario)
    logging.debug("Did not find weight labels in HEPMC file")

    return [default_weight_label]
