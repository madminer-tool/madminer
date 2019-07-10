from __future__ import absolute_import, division, print_function, unicode_literals

import os
import logging

from madminer.utils.various import call_command

logger = logging.getLogger(__name__)


def run_delphes(
    delphes_directory,
    delphes_card_filename,
    hepmc_sample_filename,
    delphes_sample_filename=None,
    initial_command=None,
    log_file=None,
    overwrite_existing_delphes_root_file=True,
    delete_unzipped_file=True,
):
    """ Runs Delphes on a HepMC sample """

    # Untar event file
    filename, extension = os.path.splitext(hepmc_sample_filename)
    to_delete = None
    if extension == ".gz":
        logger.debug("Unzipping %s", hepmc_sample_filename)
        if not os.path.exists(filename):
            call_command("gunzip -c {} > {}".format(hepmc_sample_filename, filename))
        if delete_unzipped_file:
            to_delete = filename
        hepmc_sample_filename = filename

    # Where to put Delphes sample
    if delphes_sample_filename is None:
        filename_prefix = hepmc_sample_filename.replace(".hepmc.gz", "")
        filename_prefix = filename_prefix.replace(".hepmc", "")

        for i in range(1, 1000):
            if i == 1:
                filename_candidate = filename_prefix + "_delphes.root"
            else:
                filename_candidate = filename_prefix + "_delphes_" + str(i) + ".root"

            if not os.path.exists(filename_candidate):
                delphes_sample_filename = filename_candidate
                break
            elif overwrite_existing_delphes_root_file:
                delphes_sample_filename = filename_candidate
                os.remove(delphes_sample_filename)
                break

        assert delphes_sample_filename is not None, "Could not find filename for Delphes sample"
        assert not os.path.exists(delphes_sample_filename), "Could not find filename for Delphes sample"

    # Initial commands
    if initial_command is None:
        initial_command = ""
    else:
        initial_command = initial_command + "; "

    # Call Delphes
    _ = call_command(
        "{}{}/DelphesHepMC {} {} {}".format(
            initial_command, delphes_directory, delphes_card_filename, delphes_sample_filename, hepmc_sample_filename
        ),
        log_file=log_file,
    )

    # Delete untarred file
    if to_delete is not None:
        logger.debug("Deleting %s", to_delete)
        os.remove(to_delete)

    return delphes_sample_filename
