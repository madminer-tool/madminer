from subprocess import Popen, PIPE


def call_command(cmd):
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
    out, err = proc.communicate()
    exitcode = proc.returncode

    print(out)
    print(err)
    print(exitcode)

    return exitcode, out, err