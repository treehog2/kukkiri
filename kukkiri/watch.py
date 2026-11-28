
from __future__ import annotations

import struct
from pathlib import Path
from urllib.parse import quote


def _boxes(data: bytes, start: int, end: int):
    off = start
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", data[off : off + 8])
        head = 8
        if size == 1:
            if off + 16 > end:
                return
            size, head = struct.unpack(">Q", data[off + 8 : off + 16])[0], 16
        if size < head or off + size > end:
            return
        yield off, typ, size, head
        off += size


def _video_track(data: bytes, moov: int, size: int, head: int) -> tuple[int, int] | None:
    for t_off, t_typ, t_size, t_head in _boxes(data, moov + head, moov + size):
        if t_typ != b"trak":
            continue
        track_id, timescale, video = None, None, False
        for b_off, b_typ, b_size, b_head in _boxes(data, t_off + t_head, t_off + t_size):
            if b_typ == b"tkhd":
                version = data[b_off + b_head]
                track_id = struct.unpack(">I", data[b_off + b_head + (20 if version == 1 else 12) :][:4])[0]
            elif b_typ == b"mdia":
                for m_off, m_typ, m_size, m_head in _boxes(data, b_off + b_head, b_off + b_size):
                    if m_typ == b"mdhd":
                        version = data[m_off + m_head]
                        timescale = struct.unpack(">I", data[m_off + m_head + (20 if version == 1 else 12) :][:4])[0]
                    elif m_typ == b"hdlr" and data[m_off + m_head + 8 : m_off + m_head + 12] == b"vide":
                        video = True
        if video and track_id and timescale:
            return track_id, timescale
    return None


def _fragment_time(data: bytes, moof: int, size: int, head: int, track_id: int) -> int | None:
    for t_off, t_typ, t_size, t_head in _boxes(data, moof + head, moof + size):
        if t_typ != b"traf":
            continue
        tid, time = None, None
        for b_off, b_typ, _b_size, b_head in _boxes(data, t_off + t_head, t_off + t_size):
            if b_typ == b"tfhd":
                tid = struct.unpack(">I", data[b_off + b_head + 4 : b_off + b_head + 8])[0]
            elif b_typ == b"tfdt":
                version = data[b_off + b_head]
                fmt, n = (">Q", 8) if version == 1 else (">I", 4)
                time = struct.unpack(fmt, data[b_off + b_head + 4 : b_off + b_head + 4 + n])[0]
        if tid == track_id and time is not None:
            return time
    return None


class FragmentIndex:

    def __init__(self, part: Path):
        self.part = Path(part)
        self.pos = 0
        self.init_end = None
        self.track = None
        self.frags: list[list[int]] = []
        self.ident = None
        self.first = b""

    def update(self) -> None:
        try:
            st = self.part.stat()
        except FileNotFoundError:
            self.__init__(self.part)
            return
        with open(self.part, "rb") as f:
            end = f.seek(0, 2)
            f.seek(0)
            first = f.read(64)
            if (self.ident is not None and self.ident != (st.st_ino, st.st_dev)) or end < self.pos or (
                self.first and first[: len(self.first)] != self.first
            ):
                self.__init__(self.part)
            self.ident, self.first = (st.st_ino, st.st_dev), first
            while self.pos + 16 <= end:
                f.seek(self.pos)
                head = f.read(16)
                size, typ = struct.unpack(">I4s", head[:8])
                hlen = 8
                if size == 1:
                    size, hlen = struct.unpack(">Q", head[8:16])[0], 16
                if size < hlen or self.pos + size > end:
                    return
                if typ in (b"moov", b"moof"):
                    f.seek(self.pos)
                    box = f.read(size)
                    if typ == b"moov":
                        self.init_end, self.track = self.pos + size, _video_track(box, 0, size, hlen)
                    elif self.track is not None:
                        time = _fragment_time(box, 0, size, hlen, self.track[0])
                        if time is not None:
                            self.frags.append([self.pos, size, time, 0])
                elif typ == b"mdat" and self.frags and self.frags[-1][0] + self.frags[-1][1] == self.pos:
                    self.frags[-1][1] += size
                    self.frags[-1][3] = 1
                self.pos += size

    def playlist(self, finished: bool = False) -> str | None:
        frags = [fr for fr in self.frags if fr[3]]
        if self.init_end is None or self.track is None or len(frags) < 2:
            return None
        timescale = self.track[1]
        durations = [(b[2] - a[2]) / timescale for a, b in zip(frags, frags[1:])]
        listed = frags if finished else frags[:-1]
        if finished:
            durations.append(durations[-1])
        name = quote(self.part.name)
        lines = ["#EXTM3U", "#EXT-X-VERSION:7", "#EXT-X-PLAYLIST-TYPE:EVENT",
                 f"#EXT-X-TARGETDURATION:{max(1, int(max(durations) + 0.999))}", "#EXT-X-MEDIA-SEQUENCE:0",
                 f'#EXT-X-MAP:URI="{name}",BYTERANGE="{self.init_end}@0"']
        for (off, length, _t, _c), d in zip(listed, durations):
            lines += [f"#EXTINF:{d:.3f},", f"#EXT-X-BYTERANGE:{length}@{off}", name]
        if finished:
            lines.append("#EXT-X-ENDLIST")
        return "\n".join(lines) + "\n"

    def write(self, playlist: Path, finished: bool = False) -> bool:
        try:
            self.update()
        except OSError:
            return False
        text = self.playlist(finished)
        if text is None:
            return False
        tmp = playlist.with_name(playlist.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(playlist)
        return True


def playlist_path(part: Path) -> Path:
    return Path(part).with_name(Path(part).name + ".m3u8")
