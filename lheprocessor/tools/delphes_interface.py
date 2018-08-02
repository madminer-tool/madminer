from __future__ import absolute_import, division, print_function, unicode_literals

import os

from lheprocessor.tools.utils import call_command


def run_delphes(delphes_directory,
                delphes_card_filename,
                hepmc_sample_filename,
                delphes_sample_filename=None,
                initial_command=None,
                log_file=None):

    # Untar event file
    filename, extension = os.path.splitext(hepmc_sample_filename)
    if extension == '.gz':
        if not os.path.exists(filename):
            call_command('gunzip -k {}'.format(hepmc_sample_filename))
        hepmc_sample_filename = filename

    # Where to put Delphes sample
    if delphes_sample_filename is None:
        delphes_sample_filename = None

        filename_prefix = hepmc_sample_filename.replace('.hepmc.gz', '')
        filename_prefix = filename_prefix.replace('.hepmc', '')

        for i in range(1, 1000):
            if i == 1:
                filename_candidate = filename_prefix + '_delphes.root'
            else:
                filename_candidate = filename_prefix + '_delphes_' + str(i) + '.root'

            if not os.path.exists(filename_candidate):
                delphes_sample_filename = filename_candidate
                break

        assert delphes_sample_filename is not None, "Could not find filename for Delphes sample"
        assert not os.path.exists(delphes_sample_filename), "Could not find filename for Delphes sample"

    # Initial commands
    if initial_command is None:
        initial_command = ''
    else:
        initial_command = initial_command + '; '

    # Call Delphes
    _ = call_command(
        '{}{}/DelphesHepMC {} {} {}'.format(
            initial_command, delphes_directory,
            delphes_card_filename, delphes_sample_filename, hepmc_sample_filename
        ),
        log_file=log_file
    )

    return delphes_sample_filename
