

def count_wave_params(name):
    total_parameters = 0
    for variable in name:
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    log_string("Total training params: %.1fk" % (total_parameters / 1e3))