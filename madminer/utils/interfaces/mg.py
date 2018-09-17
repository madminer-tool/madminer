from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil

from madminer.utils.various import call_command


def generate_mg_process(mg_directory,
                        temp_directory,
                        proc_card_file,
                        mg_process_directory,
                        initial_command=None,
                        log_file=None):
    # MG commands
    temp_proc_card_file = temp_directory + '/generate.mg5'
    shutil.copyfile(proc_card_file, temp_proc_card_file)

    with open(temp_proc_card_file, "a") as myfile:
        myfile.write('\n\noutput ' + mg_process_directory)

    # Call MG5
    if initial_command is None:
        initial_command = ''
    else:
        initial_command = initial_command + '; '

    _ = call_command(initial_command + mg_directory + '/bin/mg5_aMC ' + temp_proc_card_file,
                     log_file=log_file)


def run_mg_pythia(mg_directory,
                  mg_process_directory,
                  temp_directory,
                  run_card_file=None,
                  param_card_file=None,
                  reweight_card_file=None,
                  pythia8_card_file=None,
                  is_background=False,
                  initial_command=None,
                  log_file=None):
    # Remove unneeded cards
    if os.path.isfile(mg_process_directory + '/Cards/delphes_card.dat'):
        os.remove(mg_process_directory + '/Cards/delphes_card.dat')
    if os.path.isfile(mg_process_directory + '/Cards/pythia_card.dat'):
        os.remove(mg_process_directory + '/Cards/pythia_card.dat')
    if os.path.isfile(mg_process_directory + '/Cards/pythia_card.dat'):
        os.remove(mg_process_directory + '/Cards/pythia8_card.dat')
    if os.path.isfile(mg_process_directory + '/Cards/madanalysis5_hadron_card.dat'):
        os.remove(mg_process_directory + '/Cards/madanalysis5_hadron_card.dat')
    if os.path.isfile(mg_process_directory + '/Cards/madanalysis5_parton_card.dat'):
        os.remove(mg_process_directory + '/Cards/madanalysis5_parton_card.dat')

    if os.path.isfile(mg_process_directory + '/RunWeb'):
        os.remove(mg_process_directory + '/RunWeb')
    if os.path.isfile(mg_process_directory + '/index.html'):
        os.remove(mg_process_directory + '/index.html')
    if os.path.isfile(mg_process_directory + '/crossx.html'):
        os.remove(mg_process_directory + '/crossx.html')

    if os.path.isdir(mg_process_directory + '/HTML'):
        shutil.rmtree(mg_process_directory + '/HTML')
    if os.path.isdir(mg_process_directory + '/Events'):
        shutil.rmtree(mg_process_directory + '/Events')
    if os.path.isdir(mg_process_directory + '/rw_me'):
        shutil.rmtree(mg_process_directory + '/rw_me')
    if os.path.isdir(mg_process_directory + '/rw_me_second'):
        shutil.rmtree(mg_process_directory + '/rw_me_second')

    os.mkdir(mg_process_directory + '/HTML')
    os.mkdir(mg_process_directory + '/Events')

    # Copy cards
    if run_card_file is not None:
        shutil.copyfile(run_card_file, mg_process_directory + '/Cards/run_card.dat')
    if param_card_file is not None:
        shutil.copyfile(param_card_file, mg_process_directory + '/Cards/param_card.dat')
    if reweight_card_file is not None and not is_background:
        shutil.copyfile(reweight_card_file, mg_process_directory + '/Cards/reweight_card.dat')
    if pythia8_card_file is not None:
        shutil.copyfile(pythia8_card_file, mg_process_directory + '/Cards/pythia8_card.dat')

    # MG commands
    temp_proc_card_file = temp_directory + '/run.mg5'

    shower_option = 'OFF' if pythia8_card_file is None else 'Pythia8'
    reweight_option = 'OFF' if is_background else 'ON'

    mg_commands = '''
        launch {}
        shower={}
        detector=OFF
        analysis=OFF
        madspin=OFF
        reweight={}
        done
        '''.format(mg_process_directory, shower_option, reweight_option)

    with open(temp_proc_card_file, 'w') as file:
        file.write(mg_commands)

    # Call MG5
    if initial_command is None:
        initial_command = ''
    else:
        initial_command = initial_command + '; '
    _ = call_command(initial_command + mg_directory + '/bin/mg5_aMC ' + temp_proc_card_file,
                     log_file=log_file)
