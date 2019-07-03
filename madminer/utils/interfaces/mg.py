from __future__ import absolute_import, division, print_function, unicode_literals

import os
import shutil
import logging

from madminer.utils.various import call_command, make_file_executable, create_missing_folders

logger = logging.getLogger(__name__)


def generate_mg_process(
    mg_directory,
    temp_directory,
    proc_card_file,
    mg_process_directory,
    ufo_model_directory=None,
    log_file=None,
    initial_command=None,
    explicit_python_call=False,
):

    """
    Calls MadGraph to create the process folder.

    Parameters
    ----------
    mg_directory : str
        Path to the MadGraph 5 directory.

    temp_directory : str
        Path to a directory for temporary files.

    proc_card_file : str
        Path to the process card that tells MadGraph how to generate the process.

    mg_process_directory : str
        Path to the MG process directory.

    ufo_model_directory : str or None, optional
        Path to a UFO model that is not yet installed. It will be copied to the MG directory before the process card
        is executed. Default value: None.

    initial_command : str or None, optional
        Initial bash commands that have to be executed before MG is run (e.g. to load the correct virtual
        environment). Default value: None.

    log_file : str or None, optional
        Path to a log file in which the MadGraph output is saved. Default value: None.

    explicit_python_call : bool, optional
        Calls `python2.7` instead of `python`.

    Returns
    -------
        None

    """

    # Preparations
    logger.info("Generating MadGraph process folder from %s at %s", proc_card_file, mg_process_directory)

    create_missing_folders([temp_directory, mg_process_directory, os.path.dirname(log_file)])

    if ufo_model_directory is not None:
        copy_ufo_model(ufo_model_directory, mg_directory)

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

    # Explicitly call Python 2 if necessary
    python_call = "python2.7 " if explicit_python_call else ""

    _ = call_command(
        initial_command + python_call + mg_directory + "/bin/mg5_aMC " + temp_proc_card_file, log_file=log_file
    )


def setup_mg_with_scripts(
    mg_process_directory,
    proc_card_filename_from_mgprocdir=None,
    run_card_file_from_mgprocdir=None,
    param_card_file_from_mgprocdir=None,
    reweight_card_file_from_mgprocdir=None,
    pythia8_card_file_from_mgprocdir=None,
    is_background=False,
    script_file_from_mgprocdir=None,
    initial_command=None,
    log_dir=None,
    log_file_from_logdir=None,
    explicit_python_call=False,
):
    """
    Prepares a bash script that will start the event generation.

    Parameters
    ----------
    mg_process_directory : str
        Path to the MG process directory.

    proc_card_filename_from_mgprocdir : str or None, optional
        Filename for the MG command card that will be generated, relative from mg_process_directory. If None, a
        default filename in the MG process directory will be chosen.

    run_card_file_from_mgprocdir : str or None, optional
        Path to the MadGraph run card, relative from mg_process_directory. If None, the card present in the process
        folder is used. Default value: None.

    param_card_file_from_mgprocdir : str or None, optional
        Path to the MadGraph run card, relative from mg_process_directory. If None, the card present in the process
        folder is used. Default value: None.

    reweight_card_file_from_mgprocdir : str or None, optional
        Path to the MadGraph reweight card, relative from mg_process_directory. If None, the card present in the
        process folder is used. Default value: None.

    pythia8_card_file_from_mgprocdir : str or None, optional
        Path to the MadGraph Pythia8 card, relative from mg_process_directory. If None, Pythia is not run. Default
        value: None.

    is_background : bool, optional
        Should be True for background processes, i.e. process in which the differential cross section does not
        depend on the parameters (and would be the same for all benchmarks). In this case, no reweighting is run,
        which can substantially speed up the event generation. Default value: False.

    script_file_from_mgprocdir : str or None, optional
        This sets where the shell script to run MG and Pythia is generated, relative from mg_process_directory. If
        None, a default filename in `mg_process_directory/madminer` is used. Default value: None.

    initial_command : str or None, optional
        Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
        Default value: None.

    log_dir : str or None, optional
        Log directory. Default value: None.

    log_file_from_logdir : str or None, optional
        Path to a log file in which the MadGraph output is saved, relative from the default log directory. Default
        value: None.

    explicit_python_call : bool, optional
        Calls `python2.7` instead of `python`.

    Returns
    -------
    bash_script_call : str
        How to call this script.

    """

    # Preparations
    create_missing_folders([mg_process_directory])
    if log_dir is not None:
        create_missing_folders([log_dir])
    if proc_card_filename_from_mgprocdir is not None:
        create_missing_folders([os.path.dirname(mg_process_directory + "/" + proc_card_filename_from_mgprocdir)])

    # Prepare run...
    logger.info("Preparing script to run MadGraph and Pythia in %s", mg_process_directory)

    # Bash script can optionally provide MG path or process directory
    mg_directory_placeholder = "$mgdir"
    mg_process_directory_placeholder = "$mgprocdir"
    log_dir_placeholder = "$mmlogdir"
    placeholder_definition = "mgdir=$1\nmgprocdir=$2\nmmlogdir=$3"

    # Find filenames for process card and script
    if proc_card_filename_from_mgprocdir is None:
        for i in range(1000):
            proc_card_filename_from_mgprocdir = "/Cards/start_event_generation_{}.mg5".format(i)
            if not os.path.isfile(mg_process_directory + "/" + proc_card_filename_from_mgprocdir):
                break
    else:
        proc_card_filename = mg_process_directory + "/" + proc_card_filename_from_mgprocdir

    if script_file_from_mgprocdir is None:
        for i in range(1000):
            script_file = mg_process_directory + "/madminer/scripts/madminer_run_{}.sh".format(i)
            if not os.path.isfile(script_file):
                break
    else:
        script_file = mg_process_directory + "/" + script_file_from_mgprocdir

    script_filename = os.path.basename(script_file)

    if log_file_from_logdir is None:
        log_file_from_logdir = "/log.log"

    # MG commands
    shower_option = "OFF" if pythia8_card_file_from_mgprocdir is None else "Pythia8"
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
        mg_process_directory_placeholder, shower_option, reweight_option
    )

    with open(proc_card_filename, "w") as file:
        file.write(mg_commands)

    # Initial commands
    if initial_command is None:
        initial_command = ""

    #  Card copying commands
    copy_commands = ""
    if run_card_file_from_mgprocdir is not None:
        copy_commands += "cp {}/{} {}{}\n".format(
            mg_process_directory_placeholder,
            run_card_file_from_mgprocdir,
            mg_process_directory_placeholder,
            "/Cards/run_card.dat",
        )
    if param_card_file_from_mgprocdir is not None:
        copy_commands += "cp {}/{} {}{}\n".format(
            mg_process_directory_placeholder,
            param_card_file_from_mgprocdir,
            mg_process_directory_placeholder,
            "/Cards/param_card.dat",
        )
    if reweight_card_file_from_mgprocdir is not None and not is_background:
        copy_commands += "cp {}/{} {}{}\n".format(
            mg_process_directory_placeholder,
            reweight_card_file_from_mgprocdir,
            mg_process_directory_placeholder,
            "/Cards/reweight_card.dat",
        )
    if pythia8_card_file_from_mgprocdir is not None:
        copy_commands += "cp {}/{} {}{}\n".format(
            mg_process_directory_placeholder,
            pythia8_card_file_from_mgprocdir,
            mg_process_directory_placeholder,
            "/Cards/pythia8_card.dat",
        )

    # Replace environment variable in proc card
    replacement_command = """sed -e 's@\$mgprocdir@'"$mgprocdir"'@' {}/{} > {}/{}""".format(
        mg_process_directory_placeholder,
        proc_card_filename_from_mgprocdir,
        mg_process_directory_placeholder,
        "Cards/mg_commands.mg5",
    )

    # Explicitly call Python 2 if necessary
    python_call = "python2.7 " if explicit_python_call else ""

    # Put together script
    script = (
        "#!/bin/bash\n\n# Script generated by MadMiner\n\n# Usage: {} MG_directory MG_process_directory log_dir\n\n"
        + "{}\n\n{}\n\n{}\n{}\n\n{} {}/bin/mg5_aMC {}/{} > {}/{}\n"
    ).format(
        script_filename,
        initial_command,
        placeholder_definition,
        copy_commands,
        replacement_command,
        python_call,
        mg_directory_placeholder,
        mg_process_directory_placeholder,
        "Cards/mg_commands.mg5",
        log_dir_placeholder,
        log_file_from_logdir,
    )

    with open(script_file, "w") as file:
        file.write(script)
    make_file_executable(script_file)

    # How to call it from master script
    call_placeholder = "{}/{} {} {} {}".format(
        mg_process_directory_placeholder,
        script_file_from_mgprocdir,
        mg_directory_placeholder,
        mg_process_directory_placeholder,
        log_dir_placeholder,
    )

    return call_placeholder


def run_mg(
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
    explicit_python_call=False,
):
    """
    Calls MadGraph to generate events.

    Parameters
    ----------
    mg_directory : str
        Path to the MadGraph 5 base directory.

    mg_process_directory : str
        Path to the MG process directory.

    proc_card_filename : str or None, optional
        Filename for the MG command card that will be generated. If None, a default filename in the MG process
        directory will be chosen.

    run_card_file : str or None, optional
        Path to the MadGraph run card. If None, the card present in the process folder is used. Default value:
        None)

    param_card_file : str or None, optional
        Path to the MadGraph param card. If None, the card present in the process folder is used. Default value:
        None)

    reweight_card_file : str or None, optional
        Path to the MadGraph reweight card. If None, the card present in the process folder is used. (Default value
        = None)

    pythia8_card_file : str or None, optional
        Path to the MadGraph Pythia8 card. If None, Pythia is not run. Default value: None.

    is_background : bool, optional
        Should be True for background processes, i.e. process in which the differential cross section does not
        depend on the parameters (and would be the same for all benchmarks). In this case, no reweighting is run,
        which can substantially speed up the event generation. Default value: False.

    initial_command : str or None, optional
        Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
        Default value: None.

    log_file : str or None, optional
        Path to a log file in which the MadGraph output is saved. Default value: None.

    explicit_python_call : bool, optional
        Calls `python2.7` instead of `python`.

    Returns
    -------
        None

    """

    # Preparations
    create_missing_folders([mg_process_directory, os.path.dirname(log_file)])
    if proc_card_filename is not None:
        create_missing_folders([os.path.dirname(proc_card_filename)])

    # Just run it already
    logger.info("Starting MadGraph and Pythia in %s", mg_process_directory)

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

    # Python 2 support
    python_call = "python2.7 " if explicit_python_call else ""
    _ = call_command(
        initial_command + python_call + mg_directory + "/bin/mg5_aMC " + proc_card_filename, log_file=log_file
    )


def copy_ufo_model(ufo_directory, mg_directory):
    _, model_name = os.path.split(ufo_directory)
    destination = mg_directory + "/models/" + model_name

    if os.path.isdir(destination):
        return

    shutil.copytree(ufo_directory, destination)


def create_master_script(log_directory, master_script_filename, mg_directory, mg_process_directory, results):
    placeholder_definition = r"mgdir=${1:-" + mg_directory + r"}" + "\n"
    placeholder_definition += r"mgprocdir=${2:-" + mg_process_directory + r"}" + "\n"
    placeholder_definition += r"mmlogdir=${3:-" + log_directory + r"}"
    commands = "\n".join(results)
    script = (
        "#!/bin/bash\n\n# Master script to generate events for MadMiner\n\n"
        + "# Usage: run.sh [MG_directory] [MG_process_directory] [log_directory]\n\n"
        + "{}\n\n{}"
    ).format(placeholder_definition, commands)
    with open(master_script_filename, "w") as file:
        file.write(script)
    make_file_executable(master_script_filename)
