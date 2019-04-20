from __future__ import absolute_import, division, print_function, unicode_literals

import os
from io import open
import logging
from madminer.utils.various import call_command

logger = logging.getLogger(__name__)


def extract_weight_order(filename, default_weight_label=None):
    # Untar event file
    new_filename, extension = os.path.splitext(filename)
    if extension == ".gz":
        if not os.path.exists(new_filename):
            call_command("gunzip -c {} > {}".format(filename, new_filename))
        filename = new_filename

    with open(filename, encoding="latin-1") as file:
        for line in file:
            terms = line.replace('"', "").split()

            if len(terms) == 0 or terms[0] != "N":
                continue

            logger.debug("Parsing HepMC line: %s", line)

            n_benchmarks = int(terms[1])
            if not len(terms) == n_benchmarks + 2:
                logger.warning(
                    "Wrong number of weights in HepMC file. This is fine if the"
                    " weights are parsed from the LHE file, but will lead to "
                    "issues otherwise."
                )

                return None

            weight_labels = []
            for term in terms[2:]:

                if term.startswith("id="):
                    term = term[3:]
                    term = term.partition("_MERGING=")[0]
                    weight_labels.append(term)
                else:
                    weight_labels.append(default_weight_label)

            logger.debug("Found weight labels in HepMC file: %s", weight_labels)

            return weight_labels

    # Default result (no reweighting, background scenario)
    logger.debug("Did not find weight labels in HepMC file")

    return [default_weight_label]
