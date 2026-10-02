from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor


def map_tasks(func, tasks, jobs: int):
    tasks = list(tasks)
    if jobs is None or jobs <= 1:
        return [func(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(func, tasks))
