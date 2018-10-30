from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import logging

from madminer.utils.various import call_command, make_file_executable


def generate_mg_process(
    mg_directory, temp_directory, proc_card_file, mg_process_directory, initial_command=None, log_file=None
):
    """

    Parameters
    ----------
    mg_directory :
        
    temp_directory :
        
    proc_card_file :
        
    mg_process_directory :
        
    initial_command :
         (Default value = None)
    log_file :
         (Default value = None)

    Returns
    -------

    """
    # MG commands
    temp_proc_card_file = temp_directory + "/generate.mg5"
    shutil.copyfile(proc_card_file, temp_proc_card_file)

    with open(temp_proc_card_file, "a") as myfile:
        myfile.write("\n\noutput " + mg_process_directory)

    # Call MG5
    if initial_command is None:
        initial_command = ""
    else:
        initial_command = initial_command + "; "

    _ = call_command(initial_command + mg_directory + "/bin/mg5_aMC " + temp_proc_card_file, log_file=log_file)


def prepare_run_mg_pythia(
    mg_directory,
    mg_process_directory,
    proc_card_filename=None,
    run_card_file=None,
    param_card_file=None,
    reweight_card_file=None,
    pythia8_card_file=None,
    is_background=False,
    script_file=None,
    initial_command=None,
    log_file=None,
):
    """

    Parameters
    ----------
    mg_directory :
        
    mg_process_directory :
        
    proc_card_filename :
         (Default value = None)
    run_card_file :
         (Default value = None)
    param_card_file :
         (Default value = None)
    reweight_card_file :
         (Default value = None)
    pythia8_card_file :
         (Default value = None)
    is_background :
         (Default value = False)
    script_file :
         (Default value = None)
    initial_command :
         (Default value = None)
    log_file :
         (Default value = None)

    Returns
    -------

    """
    # Find filenames for process card and script
    if proc_card_filename is None:
        for i in range(1000):
            proc_card_filename = mg_process_directory + "/Cards/start_event_generation_{}.mg5".format(i)
            if not os.path.isfile(proc_card_filename):
                break

    if script_file is None:
        for i in range(1000):
            script_file = mg_process_directory + "/madminer_run_{}.sh".format(i)
            if not os.path.isfile(script_file):
                break

    # MG commands
    shower_option = "OFF" if pythia8_card_file is None else "Pythia8"
    reweight_option = "OFF" if is_background else "ON"

    mg_commands = """
        launch {}
        shower={}
        detector=OFF
        analysis=OFF
        madspin=OFF
        reweight={}
        done
        """.format(
        mg_process_directory, shower_option, reweight_option
    )

    with open(proc_card_filename, "w") as file:
        file.write(mg_commands)

    # Initial commands
    if initial_command is None:
        initial_command = ""
    else:
        initial_command = initial_command + "\n\n"

    #  Card copying commands
    copy_commands = ""
    if run_card_file is not None:
        copy_commands += "cp {} {}\n".format(run_card_file, mg_process_directory + "/Cards/run_card.dat")
    if param_card_file is not None:
        copy_commands += "cp {} {}\n".format(param_card_file, mg_process_directory + "/Cards/param_card.dat")
    if reweight_card_file is not None and not is_background:
        copy_commands += "cp {} {}\n".format(reweight_card_file, mg_process_directory + "/Cards/reweight_card.dat")
    if pythia8_card_file is not None:
        copy_commands += "cp {} {}\n".format(pythia8_card_file, mg_process_directory + "/Cards/pythia8_card.dat")

    # Put together script
    script = "#!/bin/bash\n\n{}\n\n{}\n\n{}/bin/mg5_aMC {} > {}\n".format(
        initial_command, copy_commands, mg_directory, proc_card_filename, log_file
    )

    with open(script_file, "w") as file:
        file.write(script)
    make_file_executable(script_file)

    return script_file


def run_mg_pythia(
    mg_directory,
    mg_process_directory,
    proc_card_filename=None,
    run_card_file=None,
    param_card_file=None,
    reweight_card_file=None,
    pythia8_card_file=None,
    is_background=False,
    initial_command=None,
    log_file=None,
):
    """

    Parameters
    ----------
    mg_directory :
        
    mg_process_directory :
        
    proc_card_filename :
         (Default value = None)
    run_card_file :
         (Default value = None)
    param_card_file :
         (Default value = None)
    reweight_card_file :
         (Default value = None)
    pythia8_card_file :
         (Default value = None)
    is_background :
         (Default value = False)
    initial_command :
         (Default value = None)
    log_file :
         (Default value = None)

    Returns
    -------

    """
    # # Remove unneeded cards
    # if os.path.isfile(mg_process_directory + '/Cards/delphes_card.dat'):
    #     os.remove(mg_process_directory + '/Cards/delphes_card.dat')
    # if os.path.isfile(mg_process_directory + '/Cards/pythia_card.dat'):
    #     os.remove(mg_process_directory + '/Cards/pythia_card.dat')
    # if os.path.isfile(mg_process_directory + '/Cards/pythia_card.dat'):
    #     os.remove(mg_process_directory + '/Cards/pythia8_card.dat')
    # if os.path.isfile(mg_process_directory + '/Cards/madanalysis5_hadron_card.dat'):
    #     os.remove(mg_process_directory + '/Cards/madanalysis5_hadron_card.dat')
    # if os.path.isfile(mg_process_directory + '/Cards/madanalysis5_parton_card.dat'):
    #     os.remove(mg_process_directory + '/Cards/madanalysis5_parton_card.dat')
    #
    # if os.path.isfile(mg_process_directory + '/RunWeb'):
    #     os.remove(mg_process_directory + '/RunWeb')
    # if os.path.isfile(mg_process_directory + '/index.html'):
    #     os.remove(mg_process_directory + '/index.html')
    # if os.path.isfile(mg_process_directory + '/crossx.html'):
    #     os.remove(mg_process_directory + '/crossx.html')
    #
    # if os.path.isdir(mg_process_directory + '/HTML'):
    #     shutil.rmtree(mg_process_directory + '/HTML')
    # if os.path.isdir(mg_process_directory + '/Events'):
    #     shutil.rmtree(mg_process_directory + '/Events')
    # if os.path.isdir(mg_process_directory + '/rw_me'):
    #     shutil.rmtree(mg_process_directory + '/rw_me')
    # if os.path.isdir(mg_process_directory + '/rw_me_second'):
    #     shutil.rmtree(mg_process_directory + '/rw_me_second')
    #
    # os.mkdir(mg_process_directory + '/HTML')
    # os.mkdir(mg_process_directory + '/Events')

    # Copy cards
    if run_card_file is not None:
        shutil.copyfile(run_card_file, mg_process_directory + "/Cards/run_card.dat")
    if param_card_file is not None:
        shutil.copyfile(param_card_file, mg_process_directory + "/Cards/param_card.dat")
    if reweight_card_file is not None and not is_background:
        shutil.copyfile(reweight_card_file, mg_process_directory + "/Cards/reweight_card.dat")
    if pythia8_card_file is not None:
        shutil.copyfile(pythia8_card_file, mg_process_directory + "/Cards/pythia8_card.dat")

    # Find filenames for process card and script
    if proc_card_filename is None:
        for i in range(1000):
            proc_card_filename = mg_process_directory + "/Cards/start_event_generation_{}.mg5".format(i)
            if not os.path.isfile(proc_card_filename):
                break

    # MG commands
    shower_option = "OFF" if pythia8_card_file is None else "Pythia8"
    reweight_option = "OFF" if is_background else "ON"

    mg_commands = """
        launch {}
        shower={}
        detector=OFF
        analysis=OFF
        madspin=OFF
        reweight={}
        done
        """.format(
        mg_process_directory, shower_option, reweight_option
    )

    with open(proc_card_filename, "w") as file:
        file.write(mg_commands)

    # Call MG5 or export into script
    if initial_command is None:
        initial_command = ""
    else:
        initial_command = initial_command + "; "

    _ = call_command(initial_command + mg_directory + "/bin/mg5_aMC " + proc_card_filename, log_file=log_file)


def copy_ufo_model(ufo_directory, mg_directory):
    """

    Parameters
    ----------
    ufo_directory :
        
    mg_directory :
        

    Returns
    -------

    """
    _, model_name = os.path.split(ufo_directory)
    destination = mg_directory + "/models/" + model_name

    if os.path.isdir(destination):
        return

    shutil.copytree(ufo_directory, destination)
