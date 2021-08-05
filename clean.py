import re

def regex_select(lst, regex):
    """
    Return all values from a list of strings that match any of the supplied regexes.
    """
    if isinstance(regex, str):
        regex = [regex]
        
    results = []
    for value in lst:
        for pattern in regex:
            if re.search(pattern, value):
                results.append(value)
                break
    return results