def verbose_print(verbose=True, content=None):
    if not verbose:
        return
    if content is not None:
        print(content)
    else:
        print()
