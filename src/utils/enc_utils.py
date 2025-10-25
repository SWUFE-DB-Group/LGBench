


def str_to_bytes(s):
    '''
    Convert '\\x..' string to bytes
    '''
    return s.encode('utf-8').decode('unicode_escape').encode('latin1')

