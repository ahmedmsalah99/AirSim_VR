"""2D occupancy grid built from lidar, and A* path planning on it.

No prior map: every cell starts unknown and is treated as free, so the
planner drives straight at the goal until the lidar shows something in the
way, and then replans around it. Each scan raises the log-odds of the cells
it hits and lowers those its rays pass through, so a moving thing or a
stray hit clears again.

Positions are 2D (x, y) in metres in one fixed frame (the UGV's local NED).
"""
import heapq
import math

import numpy as np

L_HIT, L_FREE, L_MIN, L_MAX = 0.85, -0.4, -2.0, 3.5
L_OCCUPIED = 0.7  # above this a cell is an obstacle: one hit is enough
SQRT2 = math.sqrt(2)
NEIGHBOURS = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0),
              (0, 1, 1.0), (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2)]


class OccupancyGrid:
    def __init__(self, center_xy, size_m, resolution_m):
        self.res = resolution_m
        self.n = int(math.ceil(size_m / resolution_m))
        self.origin = np.asarray(center_xy, float) - self.n * resolution_m / 2  # corner of cell (0, 0)
        self.logodds = np.zeros((self.n, self.n), np.float32)
        self._blocked = {}  # radius -> grown obstacles, until the next update

    # ---- cells -------------------------------------------------------------------------

    def to_cell(self, xy):
        """Cell indices (i along x, j along y) of points xy, shape (..., 2)."""
        return np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)

    def to_xy(self, cell):
        """Centre of cell (i, j)."""
        return self.origin + (np.asarray(cell, float) + 0.5) * self.res

    def inside(self, cells):
        cells = np.asarray(cells)
        return np.all((cells >= 0) & (cells < self.n), axis=-1)

    # ---- mapping -----------------------------------------------------------------------

    def update(self, sensor_xy, hits_xy):
        """One scan: hits_xy are obstacle points seen from sensor_xy. The cells between the
        sensor and each hit get freer, the hit cells more occupied (each once per scan)."""
        hits = np.asarray(hits_xy, float).reshape(-1, 2)
        if not len(hits):
            return
        # One ray per hit cell, not per point: a lidar sweep has thousands of points, mostly
        # several to a cell, and every ray is sampled along its whole length.
        cells = self._unique_cells(hits)
        if not len(cells):
            return
        hits = self.to_xy(cells)
        sensor = np.asarray(sensor_xy, float)
        d = np.linalg.norm(hits - sensor, axis=1)
        step = self.res / 2
        steps = np.arange(0, d.max(), step)
        # Samples along every ray, stopping a cell short of the hit.
        t = steps[None, :] / np.maximum(d[:, None], 1e-9)
        keep = steps[None, :] < d[:, None] - self.res
        samples = sensor + t[..., None] * (hits - sensor)[:, None, :]
        free = self._unique_cells(samples[keep])
        occupied = self._unique_cells(hits)
        if len(free) and len(occupied):
            # A cell hit in this scan isn't also freed by another ray passing through it.
            free = free[~np.isin(free[:, 0] * self.n + free[:, 1], occupied[:, 0] * self.n + occupied[:, 1])]
        if len(free):
            self.logodds[free[:, 0], free[:, 1]] += L_FREE
        if len(occupied):
            self.logodds[occupied[:, 0], occupied[:, 1]] += L_HIT
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)
        self._blocked = {}

    def _unique_cells(self, xy):
        cells = self.to_cell(xy)
        cells = cells[self.inside(cells)]
        return np.unique(cells, axis=0) if len(cells) else cells

    def occupied(self):
        return self.logodds > L_OCCUPIED

    def blocked(self, radius_m):
        """Occupied cells grown by radius_m: where the robot's centre can't go (a copy)."""
        if radius_m not in self._blocked:
            self._blocked[radius_m] = self._grow(radius_m)
        return self._blocked[radius_m].copy()

    def _grow(self, radius_m):
        """Occupied cells and every cell within radius_m of one (obstacles are sparse, so the
        disk is stamped around each occupied cell rather than shifting the whole map)."""
        out = np.zeros((self.n, self.n), bool)
        cells = np.argwhere(self.occupied())
        if not len(cells):
            return out
        r = int(math.ceil(radius_m / self.res))
        di, dj = np.mgrid[-r:r + 1, -r:r + 1]
        disk = np.stack([di[di ** 2 + dj ** 2 <= r * r], dj[di ** 2 + dj ** 2 <= r * r]], axis=1)
        grown = (cells[:, None, :] + disk[None, :, :]).reshape(-1, 2)
        grown = grown[self.inside(grown)]
        out[grown[:, 0], grown[:, 1]] = True
        return out

    # ---- planning ----------------------------------------------------------------------

    def plan(self, start_xy, goal_xy, radius_m, goal_tolerance_m=0.0, max_expansions=150_000):
        """Shortest collision-free path from start to within goal_tolerance_m of goal, as a list of
        (x, y) waypoints (start first), or None. Unknown cells count as free. The cells within
        radius_m of the start are freed, so a robot that has drifted close to an obstacle can
        still leave."""
        blocked = self.blocked(radius_m)
        start, goal = tuple(self.to_cell(start_xy)), tuple(self.to_cell(goal_xy))
        if not (self.inside(start) and self.inside(goal)):
            return None
        r = int(math.ceil(radius_m / self.res))
        si, sj = start
        blocked[max(si - r, 0):si + r + 1, max(sj - r, 0):sj + r + 1] = False
        tol = goal_tolerance_m / self.res
        gi, gj = goal

        def h(i, j):
            return max(math.hypot(i - gi, j - gj) - tol, 0.0)

        g = {start: 0.0}
        parent = {start: None}
        heap = [(h(*start), 0.0, start)]
        done = set()
        expansions = 0
        while heap:
            _, cost, cell = heapq.heappop(heap)
            if cell in done:
                continue
            done.add(cell)
            i, j = cell
            if math.hypot(i - gi, j - gj) <= tol + 0.5:
                return self._smooth(self._path(parent, cell), blocked)
            expansions += 1
            if expansions > max_expansions:
                return None
            for di, dj, c in NEIGHBOURS:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.n and 0 <= nj < self.n) or blocked[ni, nj]:
                    continue
                if di and dj and (blocked[i + di, j] or blocked[i, j + dj]):
                    continue  # no corner cutting
                new = cost + c
                if new < g.get((ni, nj), math.inf):
                    g[(ni, nj)] = new
                    parent[(ni, nj)] = cell
                    heapq.heappush(heap, (new + h(ni, nj), new, (ni, nj)))
        return None

    def _path(self, parent, cell):
        cells = []
        while cell is not None:
            cells.append(cell)
            cell = parent[cell]
        return cells[::-1]

    def _smooth(self, cells, blocked):
        """Drop waypoints the robot can skip in a straight line."""
        if len(cells) <= 2:
            return [tuple(self.to_xy(c)) for c in cells]
        out = [cells[0]]
        k = 0
        while k < len(cells) - 1:
            far = len(cells) - 1
            while far > k + 1 and not self._line_free(cells[k], cells[far], blocked):
                far -= 1
            out.append(cells[far])
            k = far
        return [tuple(self.to_xy(c)) for c in out]

    def _line_free(self, a, b, blocked):
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1])) * 2) + 1
        i = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        j = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        return not blocked[i, j].any()

    def path_blocked(self, path, radius_m):
        """Whether any segment of path now crosses an obstacle grown by radius_m (skipping the
        first cells, around the robot)."""
        if not path or len(path) < 2:
            return False
        blocked = self.blocked(radius_m)
        r = int(math.ceil(radius_m / self.res))
        si, sj = self.to_cell(path[0])
        blocked[max(si - r, 0):si + r + 1, max(sj - r, 0):sj + r + 1] = False
        cells = [tuple(self.to_cell(p)) for p in path]
        if not all(self.inside(c) for c in cells):
            return True
        return any(not self._line_free(a, b, blocked) for a, b in zip(cells, cells[1:]))
