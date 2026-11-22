from kukkiri import staged
from kukkiri.streaming import JobSource


class _Frames:

    def __init__(self, n, batch=4):
        self.n, self.batch, self.released = n, batch, []

    def __len__(self):
        return self.n

    def iter_batches(self):
        for i in range(0, self.n, self.batch):
            yield [None] * min(self.batch, self.n - i)

    def release(self, i):
        self.released.append(i)


class _Track:
    def __init__(self, tid):
        self.track_id, self.boxes = tid, {}


def test_record_then_replay_issues_the_same_jobs_at_the_same_frames():
    n = 10
    jobs = {3: [(9, _Track(1), [0, 1, 2, 3], {})], 7: [(9, _Track(2), [4, 5, 6, 7], {})]}

    def produce(src):
        for fi in range(n):
            for j in jobs.get(fi, []):
                src.put(j)
            src.advance(fi - 1)
        src.close(n)

    frames = _Frames(n)
    events = staged.plan(produce, frames, n)
    assert frames.released == list(range(n))

    seen = []

    class _Source(JobSource):
        def put(self, job):
            seen.append(("job", job[1].track_id, self.frontier))
            super().put(job)

        def advance(self, f):
            seen.append(("advance", f))
            super().advance(f)

    src = _Source(n)
    staged.replay(events, _Frames(n), src)
    assert [e[0] for e in seen] == [k for k, _ in events if k != "close"]
    assert src.frontier == n and src.n_jobs == 2
