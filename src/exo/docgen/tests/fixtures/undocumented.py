def multiply(a, b):
    return a * b


class Counter:
    def __init__(self, start=0):
        self.count = start

    def increment(self):
        self.count += 1
        return self.count
