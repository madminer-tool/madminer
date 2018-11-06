from __future__ import absolute_import, division, print_function, unicode_literals
import six


def export_param_card(benchmark, parameters, param_card_template_file, mg_process_directory, param_card_filename=None):
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
            variables = {"theta": parameter_value}
            parameter_value = eval(parameter_transform, variables)

        block_begin = param_card.lower().find(("Block " + parameter_lha_block).lower())
        if block_begin < 0:
            raise ValueError("Could not find block {0} in param_card template!".format(parameter_lha_block))

        block_end = param_card.lower().find("Block".lower(), block_begin + 5)
        if block_end < 0:
            block_end = len(param_card)

        block = param_card[block_begin:block_end].split("\n")
        changed_line = False
        for i, line in enumerate(block):
            comment_pos = line.find("#")
            if i >= 0:
                line = line[:comment_pos]
            line = line.strip()
            elements = line.split()
            if len(elements) >= 2:
                try:
                    if int(elements[0]) == parameter_lha_id:
                        block[i] = "    " + str(parameter_lha_id) + "    " + str(parameter_value) + "    # MadMiner"
                        changed_line = True
                        break
                except ValueError:
                    pass

        if not changed_line:
            raise ValueError("Could not find LHA ID {0} in param_card template!".format(parameter_lha_id))

        param_card = param_card[:block_begin] + "\n".join(block) + param_card[block_end:]

    # Output filename
    if param_card_filename is None:
        param_card_filename = mg_process_directory + "/Cards/param_card.dat"

    # Save param_card.dat
    with open(param_card_filename, "w") as file:
        file.write(param_card)


def export_reweight_card(
    sample_benchmark,
    benchmarks,
    parameters,
    reweight_card_template_file,
    mg_process_directory,
    reweight_card_filename=None,
):
    # Open reweight_card template
    with open(reweight_card_template_file) as file:
        reweight_card = file.read()

    # Put in parameter values
    block_end = reweight_card.find("# Manual")
    assert block_end >= 0, 'Cannot find "# Manual" string in reweight_card template'

    insert_pos = reweight_card.rfind("\n\n", 0, block_end)
    assert insert_pos >= 0, "Cannot find empty line in reweight_card template"

    lines = []
    for benchmark_name, benchmark in six.iteritems(benchmarks):
        if benchmark_name == sample_benchmark:
            continue

        lines.append("")
        lines.append("# MadMiner benchmark " + benchmark_name)
        lines.append("launch --rwgt_name=" + benchmark_name)

        for parameter_name, parameter_value in six.iteritems(benchmark):
            parameter_lha_block = parameters[parameter_name][0]
            parameter_lha_id = parameters[parameter_name][1]

            # Transform parameters if needed
            parameter_transform = parameters[parameter_name][4]
            if parameter_transform is not None:
                variables = {"theta": parameter_value}
                parameter_value = eval(parameter_transform, variables)

            lines.append("  set {0} {1} {2}".format(parameter_lha_block, parameter_lha_id, parameter_value))

        lines.append("")

    reweight_card = reweight_card[:insert_pos] + "\n".join(lines) + reweight_card[insert_pos:]

    # Output filename
    if reweight_card_filename is None:
        reweight_card_filename = mg_process_directory + "/Cards/reweight_card.dat"

    # Save param_card.dat
    with open(reweight_card_filename, "w") as file:
        file.write(reweight_card)
