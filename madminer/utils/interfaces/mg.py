import logging
import shutil

from pathlib import Path

from madminer.utils.various import call_command
from madminer.utils.various import make_file_executable

logger = logging.getLogger(__name__)


def generate_mg_process(
    mg_directory,
    temp_directory,
    proc_card_file,
    mg_process_directory,
    ufo_model_directory=None,
    log_file=None,
    initial_command=None,
    python_executable=None,
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

    python_executable : None or str, optional
        Overwrites the default Python executable

    Returns
    -------
        None

    """

    # Preparations
    logger.info("Generating MadGraph process folder from %s at %s", proc_card_file, mg_process_directory)

    Path(temp_directory).mkdir(parents=True, exist_ok=True)
    Path(mg_process_directory).mkdir(parents=True, exist_ok=True)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    if ufo_model_directory is not None:
        copy_ufo_model(ufo_model_directory, mg_directory)

    # MG commands
    temp_proc_card_file = Path(temp_directory, "generate.mg5")
    shutil.copyfile(proc_card_file, temp_proc_card_file)

    with open(temp_proc_card_file, "a") as myfile:
        myfile.write(f"\n")
        myfile.write(f"\n")
        myfile.write(f"output {mg_process_directory}")

    # Call specific initial command and Python binary
    initial_command = f"{initial_command}; " if initial_command is not None else ""
    python_binary = f"{python_executable} " if python_executable is not None else ""

    command = f"{initial_command}{python_binary}{mg_directory}/bin/mg5_aMC {temp_proc_card_file}"
    logger.info(f"Calling MadGraph: {command}")

    _ = call_command(cmd=command, log_file=log_file)


def setup_mg_with_scripts(
    mg_process_directory,
    proc_card_filename_from_mgprocdir=None,
    run_card_file_from_mgprocdir=None,
    param_card_file_from_mgprocdir=None,
    reweight_card_file_from_mgprocdir=None,
    pythia8_card_file_from_mgprocdir=None,
    configuration_file_from_mgprocdir=None,
    is_background=False,
    script_file_from_mgprocdir=None,
    initial_command=None,
    log_dir=None,
    log_file_from_logdir=None,
    order="LO",
    python_executable=None,
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

    configuration_file_from_mgprocdir : str or None, optional
        Path to the MadGraph me5_configuration card, relative from mg_process_directory. If None, the card
        present in the process folder is used. Default value: None.

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

    python_executable : None or str, optional
        Overwrites the default Python executable

    Returns
    -------
    bash_script_call : str
        How to call this script.

    """

    # Preparations
    Path(mg_process_directory).mkdir(parents=True, exist_ok=True)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    if proc_card_filename_from_mgprocdir is not None:
        proc_path = Path(mg_process_directory, proc_card_filename_from_mgprocdir)
        proc_path.parent.mkdir(parents=True, exist_ok=True)

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
            proc_card_filename_from_mgprocdir = f"Cards/start_event_generation_{i}.mg5"
            if not Path(mg_process_directory, proc_card_filename_from_mgprocdir).is_file():
                break
    else:
        proc_card_filename = Path(mg_process_directory, proc_card_filename_from_mgprocdir)

    if script_file_from_mgprocdir is None:
        for i in range(1000):
            script_file = f"{mg_process_directory}/madminer/scripts/madminer_run_{i}.sh"
            if not Path(script_file).is_file():
                break
    else:
        script_file = f"{mg_process_directory}/{script_file_from_mgprocdir}"

    script_filename = Path(script_file).name

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

    #  Card copying commands
    copy_commands = ""
    if run_card_file_from_mgprocdir is not None:
        copy_commands += (
            f"cp "
            f"{mg_process_directory_placeholder}/{run_card_file_from_mgprocdir} "
            f"{mg_process_directory_placeholder}/Cards/run_card.dat\n"
        )
    if param_card_file_from_mgprocdir is not None:
        copy_commands += (
            f"cp "
            f"{mg_process_directory_placeholder}/{param_card_file_from_mgprocdir} "
            f"{mg_process_directory_placeholder}/Cards/param_card.dat\n"
        )
    if reweight_card_file_from_mgprocdir is not None and not is_background:
        copy_commands += (
            f"cp "
            f"{mg_process_directory_placeholder}/{reweight_card_file_from_mgprocdir} "
            f"{mg_process_directory_placeholder}/Cards/reweight_card.dat\n"
        )
    if pythia8_card_file_from_mgprocdir is not None and order == "LO":
        copy_commands += (
            f"cp "
            f"{mg_process_directory_placeholder}/{pythia8_card_file_from_mgprocdir} "
            f"{mg_process_directory_placeholder}/Cards/pythia8_card.dat\n"
        )
    if pythia8_card_file_from_mgprocdir is not None and order == "NLO":
        copy_commands += (
            f"cp "
            f"{mg_process_directory_placeholder}/{pythia8_card_file_from_mgprocdir} "
            f"{mg_process_directory_placeholder}/Cards/shower_card.dat\n"
        )
    if configuration_file_from_mgprocdir is not None:
        copy_commands += (
            f"cp "
            f"{mg_process_directory_placeholder}/{configuration_file_from_mgprocdir} "
            f"{mg_process_directory_placeholder}/Cards/me5_configuration.txt\n"
        )

    # Replace environment variable in proc card
    replacement_command = """sed -e 's@\$mgprocdir@'"$mgprocdir"'@' {}/{} > {}/{}""".format(
        mg_process_directory_placeholder,
        proc_card_filename_from_mgprocdir,
        mg_process_directory_placeholder,
        "Cards/mg_commands.mg5",
    )

    # Call specific initial command and Python binary
    initial_command = f"{initial_command} " if initial_command is not None else ""
    python_binary = f"{python_executable} " if python_executable is not None else ""

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
        python_binary,
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
    call_placeholder = (
        f"{mg_process_directory_placeholder}/{script_file_from_mgprocdir} "
        f"{mg_directory_placeholder} "
        f"{mg_process_directory_placeholder} "
        f"{log_dir_placeholder}"
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
    configuration_card_file=None,
    is_background=False,
    initial_command=None,
    log_file=None,
    order="LO",
    python_executable=None,
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

    configuration_card_file : str or None, optional
        Path to the MadGraph configuration card. If None, the card present in the process folder is used. (Default
        value: None).

    is_background : bool, optional
        Should be True for background processes, i.e. process in which the differential cross section does not
        depend on the parameters (and would be the same for all benchmarks). In this case, no reweighting is run,
        which can substantially speed up the event generation. Default value: False.

    initial_command : str or None, optional
        Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
        Default value: None.

    log_file : str or None, optional
        Path to a log file in which the MadGraph output is saved. Default value: None.

    python_executable : None or str, optional
        Overwrites the default Python executable

    Returns
    -------
        None

    """

    # Preparations
    Path(mg_process_directory).mkdir(parents=True, exist_ok=True)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    if proc_card_filename is not None:
        Path(proc_card_filename).parent.mkdir(parents=True, exist_ok=True)

    # Just run it already
    logger.info("Starting MadGraph and Pythia in %s", mg_process_directory)

    # Copy cards
    if run_card_file is not None:
        shutil.copyfile(run_card_file, f"{mg_process_directory}/Cards/run_card.dat")
    if param_card_file is not None:
        shutil.copyfile(param_card_file, f"{mg_process_directory}/Cards/param_card.dat")
    if reweight_card_file is not None and not is_background:
        shutil.copyfile(reweight_card_file, f"{mg_process_directory}/Cards/reweight_card.dat")
    if pythia8_card_file is not None and order == "LO":
        shutil.copyfile(pythia8_card_file, f"{mg_process_directory}/Cards/pythia8_card.dat")
    if pythia8_card_file is not None and order == "NLO":
        shutil.copyfile(pythia8_card_file, f"{mg_process_directory}/Cards/shower_card.dat")
    if configuration_card_file is not None:
        shutil.copyfile(configuration_card_file, f"{mg_process_directory}/Cards/me5_configuration.txt")

    # Find filenames for process card and script
    if proc_card_filename is None:
        for i in range(1000):
            proc_card_filename = f"{mg_process_directory}/Cards/start_event_generation_{i}.mg5"
            if not Path(proc_card_filename).is_file():
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

    # Call specific initial command and Python binary
    initial_command = f"{initial_command}; " if initial_command is not None else ""
    python_binary = f"{python_executable} " if python_executable is not None else ""

    command = f"{initial_command}{python_binary}{mg_directory}/bin/mg5_aMC {proc_card_filename}"
    logger.info(f"Calling MadGraph: {command}")

    _ = call_command(cmd=command, log_file=log_file)


def setup_mg_reweighting_with_scripts(
    mg_process_directory,
    run_name,
    reweight_card_file_from_mgprocdir=None,
    script_file_from_mgprocdir=None,
    initial_command=None,
    log_dir=None,
    log_file_from_logdir=None,
):
    """
    Prepares a bash script that will start the event generation.

    Parameters
    ----------
    mg_process_directory : str
        Path to the MG process directory.

    reweight_card_file_from_mgprocdir : str or None, optional
        Path to the MadGraph reweight card, relative from mg_process_directory. If None, the card present in the
        process folder is used. Default value: None.

    script_file_from_mgprocdir : str or None, optional
        This sets where the shell script to run the reweighting is generated, relative from mg_process_directory. If
        None, a default filename in `mg_process_directory/madminer` is used. Default value: None.

    initial_command : str or None, optional
        Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
        Default value: None.

    log_dir : str or None, optional
        Log directory. Default value: None.

    log_file_from_logdir : str or None, optional
        Path to a log file in which the MadGraph output is saved, relative from the default log directory. Default
        value: None.

    Returns
    -------
    bash_script_call : str
        How to call this script.

    """

    # Preparations
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Prepare run...
    logger.info("Preparing script to reweight an existing sample in %s", mg_process_directory)

    # Bash script can optionally provide MG path or process directory
    mg_process_directory_placeholder = "$mgprocdir"
    log_dir_placeholder = "$mmlogdir"
    placeholder_definition = r"mgprocdir=${1:-" + mg_process_directory + r"}" + "\n"
    placeholder_definition += r"mmlogdir=${2:-" + log_dir + r"}"

    if script_file_from_mgprocdir is None:
        script_file = f"{mg_process_directory}/madminer/scripts/madminer_reweight_{run_name}.sh"
    else:
        script_file = f"{mg_process_directory}/{script_file_from_mgprocdir}"

    script_filename = Path(script_file).name

    if log_file_from_logdir is None:
        log_file_from_logdir = "/log.log"

    # Initial commands
    if initial_command is None:
        initial_command = ""

    #  Card copying commands
    if reweight_card_file_from_mgprocdir is not None:
        copy_commands = (
            f"cp "
            f"{mg_process_directory_placeholder}/{reweight_card_file_from_mgprocdir} "
            f"{mg_process_directory_placeholder}/Cards/reweight_card.dat\n"
        )
    else:
        copy_commands = ""

    # Put together script
    script = (
        "#!/bin/bash\n\n# Script generated by MadMiner\n\n# Usage: {} MG_process_directory log_dir\n\n"
        + "{}\n\n{}\n\n{}\n\n{}/bin/madevent reweight {} -f > {}/{}\n"
    ).format(
        script_filename,
        initial_command,
        placeholder_definition,
        copy_commands,
        mg_process_directory_placeholder,
        run_name,
        log_dir_placeholder,
        log_file_from_logdir,
    )

    with open(script_file, "w") as file:
        file.write(script)
    make_file_executable(script_file)

    # How to call it from master script
    call_instruction = f"{mg_process_directory}/{script_file_from_mgprocdir} [MG_process_directory] [log_directory]"

    return call_instruction


def run_mg_reweighting(mg_process_directory, run_name, reweight_card_file=None, initial_command=None, log_file=None):
    """
    Runs MG reweighting.

    Parameters
    ----------
    mg_process_directory : str
        Path to the MG process directory.

    run_name : str
        Run name.

    reweight_card_file : str or None, optional
        Path to the MadGraph reweight card. If None, the card present in the process folder is used. (Default value
        = None)

    initial_command : str or None, optional
        Initial shell commands that have to be executed before MG is run (e.g. to load a virtual environment).
        Default value: None.

    log_file : str or None, optional
        Path to a log file in which the MadGraph output is saved. Default value: None.

    Returns
    -------
    bash_script_call : str
        How to call this script.

    """

    # Preparations
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # Prepare run...
    logger.info("Starting reweighting of an existing sample in %s", mg_process_directory)

    #  Copy cards
    if reweight_card_file is not None:
        shutil.copyfile(reweight_card_file, f"{mg_process_directory}/Cards/reweight_card.dat")

    # Call MG5 reweight feature
    initial_command = f"{initial_command}; " if initial_command else ""

    _ = call_command(
        cmd=f"{initial_command}{mg_process_directory}/bin/madevent reweight {run_name} -f",
        log_file=log_file,
    )


def copy_ufo_model(ufo_directory, mg_directory):
    model_name = Path(ufo_directory).name
    destination = Path(mg_directory, "models", model_name)

    if destination.is_dir():
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
