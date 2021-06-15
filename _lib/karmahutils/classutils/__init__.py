version_number = '0.1.3'
version = 'moonshade'


def stringify_object(object_instance):
    str_rep = """<""" + object_instance.__class__.__name__ + """ Object>
    """
    dict_rep = object_instance.__dict__
    str_rep += ','.join([X + ':' + str(dict_rep[X]) for X in dict_rep.keys()])
    return str_rep


print('loaded karmah class utils', version, version_number)
