def getRegex(text):
    if text is None:
        return None
    text = str(text)
    regex = ""
    alpha_data = False
    numeric_data = False
    for char in text:
        if char.isalpha():
            if not alpha_data:
                numeric_data = False
                regex += "[a-zA-Z]"
                alpha_data = True
        elif char.isnumeric():
            if not numeric_data:
                alpha_data = False
                regex += "[0-9]"
                numeric_data = True
        else:
            numeric_data = False
            alpha_data = False
            regex += char

    return regex

def validate_regex(regex, text):
    print(re.search(regex, text))