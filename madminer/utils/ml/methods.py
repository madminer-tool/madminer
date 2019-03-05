from __future__ import absolute_import, division, print_function

from collections import OrderedDict


def package_training_data(method, x, theta0, theta1, y, r_xz, t_xz0, t_xz1):
    method_type = get_method_type(method)
    data = OrderedDict()
    if method_type == "parameterized":
        data["x"] = x
        data["theta"] = theta0
        data["y"] = y
        if r_xz is not None:
            data["r_xz"] = r_xz
        if t_xz0 is not None:
            data["t_xz"] = t_xz0
    elif method_type == "doubly_parameterized":
        data["x"] = x
        data["theta0"] = theta0
        data["theta1"] = theta1
        data["y"] = y
        if r_xz is not None:
            data["r_xz"] = r_xz
        if t_xz0 is not None:
            data["t_xz0"] = t_xz0
        if t_xz1 is not None:
            data["t_xz1"] = t_xz1
    elif method_type == "local_score":
        data["x"] = x
        data["t_xz"] = t_xz0
    elif method_type == "nde":
        data["x"] = x
        data["theta"] = theta0
        if t_xz0 is not None:
            data["t_xz"] = t_xz0
    return data
