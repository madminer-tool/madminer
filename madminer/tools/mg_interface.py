from __future__ import absolute_import, division, print_function, unicode_literals
import six

import os

import shutil
from madminer.tools.utils import call_command


def export_param_card(benchmark,
                      parameters,
                      param_card_template_file,
                      mg_process_directory):
    # Open parameter card template
    with open(param_card_template_file) as file:
        param_card = file.read()

    # Replace parameter values
    for parameter_name, parameter_value in six.iteritems(benchmark):
        parameter_lha_block = parameters[parameter_name][0]
        parameter_lha_id = parameters[parameter_name][1]

        # Transform parameters if needed
        parameter_transform = parameters[parameter_name][4]
        if parameter_transform is not None:
            variables = {'theta': parameter_value}
            parameter_value = eval(parameter_transform, variables)

        block_begin = param_card.lower().find(('Block ' + parameter_lha_block).lower())
        if block_begin < 0:
            raise ValueError('Could not find block {0} in param_card template!'.format(parameter_lha_block))

        block_end = param_card.lower().find('Block'.lower(), block_begin + 5)
        if block_end < 0:
            block_end = len(param_card)

        block = param_card[block_begin:block_end].split('\n')
        changed_line = False
        for i, line in enumerate(block):
            comment_pos = line.find('#')
            if i >= 0:
                line = line[:comment_pos]
            line = line.strip()
            elements = line.split()
            if len(elements) >= 2:
                try:
                    if int(elements[0]) == parameter_lha_id:
                        block[i] = '    ' + str(parameter_lha_id) + '    ' + str(parameter_value) + '    # MadMiner'
                        changed_line = True
                        break
                except ValueError:
                    pass

        if not changed_line:
            raise ValueError('Could not find LHA ID {0} in param_card template!'.format(parameter_lha_id))

        param_card = param_card[:block_begin] + '\n'.join(block) + param_card[block_end:]

    # Save param_card.dat
    with open(mg_process_directory + '/Cards/param_card.dat', 'w') as file:
        file.write(param_card)


def export_reweight_card(sample_benchmark,
                         benchmarks,
                         parameters,
                         reweight_card_template_file,
                         mg_process_directory):
    # Open reweight_card template
    with open(reweight_card_template_file) as file:
        reweight_card = file.read()

    # Put in parameter values
    block_end = reweight_card.find('# Manual')
    assert block_end >= 0, 'Cannot find "# Manual" string in reweight_card template'

    insert_pos = reweight_card.rfind('\n\n', 0, block_end)
    assert insert_pos >= 0, 'Cannot find empty line in reweight_card template'

    lines = []
    for benchmark_name, benchmark in six.iteritems(benchmarks):
        if benchmark_name == sample_benchmark:
            continue

        lines.append('')
        lines.append('# MadMiner benchmark ' + benchmark_name)
        lines.append('launch --rwgt_name=' + benchmark_name)

        for parameter_name, parameter_value in six.iteritems(benchmark):
            parameter_lha_block = parameters[parameter_name][0]
            parameter_lha_id = parameters[parameter_name][1]

            # Transform parameters if needed
            parameter_transform = parameters[parameter_name][4]
            if parameter_transform is not None:
                variables = {'theta': parameter_value}
                parameter_value = eval(parameter_transform, variables)

            lines.append('  set {0} {1} {2}'.format(parameter_lha_block, parameter_lha_id, parameter_value))

        lines.append('')

    reweight_card = reweight_card[:insert_pos] + '\n'.join(lines) + reweight_card[insert_pos:]

    # Save param_card.dat
    with open(mg_process_directory + '/Cards/reweight_card.dat', 'w') as file:
        file.write(reweight_card)


# Everything below from Felix

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

