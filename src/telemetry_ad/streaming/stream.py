from collections import deque


class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self._buf = deque(maxlen=size)

    def push(self, row):
        self._buf.append(row)

    def ready(self) -> bool:
        return len(self._buf) == self.size

    def window(self):
        return list(self._buf)
