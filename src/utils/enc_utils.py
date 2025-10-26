import re


def str_to_bytes(s):
    '''
    Convert '\\x..' string to bytes
    '''
    return s.encode('utf-8').decode('unicode_escape').encode('latin1')

def is_stru_val(b:bytes, enc):
    try:
        b.decode(enc)
        return True
    except:
        return False


def is_bytes_str(s:str):
    """
    Check if the string is in escaped byte format like "\\xcf\\xc2\\xb5\\xa5"
    """

    pattern = r'^(\\x[0-9a-fA-F]{2})+$'
    return bool(re.match(pattern, s))

def remove_non_alphanumeric(s:str):
    """Remove all non-alphanumeric characters"""
    s = s.upper()
    return re.sub(r'[^\w]', '', s)


if __name__ == '__main__':
    print(is_bytes_str("\\xcf\\xc2\\xb5\\xa5"))
