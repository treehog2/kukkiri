

class FFmpegError(Exception):
    pass


class SourceTruncated(FFmpegError):

    def __init__(self, decoded: int, expected: int, fps: float):
        self.decoded, self.expected, self.fps = decoded, expected, fps
        at = f"{int(decoded / fps // 60)}:{int(decoded / fps % 60):02d}" if fps else f"frame {decoded}"
        super().__init__(f"source unreadable after {at} ({decoded} of {expected} frames decoded)")


class AlreadyRemasteredError(Exception):
    pass


class NotRemasterableError(Exception):
    pass
