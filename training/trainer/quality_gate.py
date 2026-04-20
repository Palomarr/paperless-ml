def quick_sanity_check(loss):
    if loss > 100:
        return False, "Loss exploded"
    return True, "OK"