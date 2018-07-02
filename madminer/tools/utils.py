from __future__ import absolute_import, division, print_function, unicode_literals

import os
from subprocess import Popen, PIPE
import io
import numpy as np


def normalize_xsecs(weights):
    return weights / np.sum(weights, axis=0)


def call_command(cmd, log_file=None):
    if log_file is not None:
        with io.open(log_file, 'wb') as log:
            proc = Popen(cmd, stdout=log, stderr=log, shell=True)
            _ = proc.communicate()
            exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(
                'Calling command {} returned exit code {}. Output in file {}.'.format(
                    cmd, exitcode, log_file
                )
            )
    else:
        proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        out, err = proc.communicate()
        exitcode = proc.returncode

        if exitcode != 0:
            raise RuntimeError(
                'Calling command {} returned exit code {}.\n\nStd output:\n\n{}Error output:\n\n{}'.format(
                    cmd, exitcode, out, err
                )
            )

    return exitcode


def create_missing_folders(folders):
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

        elif not os.path.isdir(folder):
            raise OSError('Path {} exists, but is no directory!'.format(folder))