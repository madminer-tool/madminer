from __future__ import absolute_import, division, print_function, unicode_literals

from subprocess import Popen, PIPE
import io
import logging


def general_init(debug=False):
    logging.basicConfig(format='%(asctime)s  %(message)s', datefmt='%H:%M')
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    logging.info('')
    logging.info('------------------------------------------------------------')
    logging.info('|                                                          |')
    logging.info('|  DelphesProcessor                                        |')
    logging.info('|                                                          |')
    logging.info('|  Version from August 13, 2018                            |')
    logging.info('|                                                          |')
    logging.info('|           Johann Brehmer, Kyle Cranmer, and Felix Kling  |')
    logging.info('|                                                          |')
    logging.info('------------------------------------------------------------')
    logging.info('')

    logging.info('Hi! How are you today?')

    # np.seterr(divide='ignore', invalid='ignore')
    # np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})


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
