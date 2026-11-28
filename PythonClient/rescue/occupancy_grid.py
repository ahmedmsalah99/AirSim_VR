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
REGROW_EVERY = 10  # scans between full rebuilds of the grown obstacles
FREE_RANGE_M = 15.0  # rays clear the cells they cross up to this far (beyond, it's slow and coarse)
SQRT2 = math.sqrt(2)
NEIGHBOURS = [(-1, -1, SQRT2), (-1, 0, 1.0), (-1, 1, SQRT2), (0, -1, 1.0),
              (0, 1, 1.0), (1, -1, SQRT2), (1, 0, 1.0), (1, 1, SQRT2)]


class OccupancyGrid:
    def __init__(self, center_xy, size_m, resolution_m):
        self.res = resolution_m
        self.n = int(math.ceil(size_m / resolution_m))
        self.origin = np.asarray(center_xy, float) - self.n * resolution_m / 2  # corner of cell (0, 0)
        self.logodds = np.zeros((self.n, self.n), np.float32)
        self._blocked = {}  # radius -> grown obstacles (kept up to date by update())
        self._updates = 0
        # Terrain: the lowest surface height (m, up) seen in each cell, and cells where the ground
        # is too steep or has a step (set by update_ground).
        self.ground = np.full((self.n, self.n), np.nan, np.float32)
        self.steep = np.zeros((self.n, self.n), bool)
        self.bounds = None  # (lo_xy, hi_xy): the robot's centre stays inside

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

    def update(self, sensor_xy, hits_xy, free_range_m=FREE_RANGE_M):
        """One scan: hits_xy are obstacle points seen from sensor_xy. The cells between the
        sensor and each hit (up to free_range_m out) get freer, the hit cells more occupied
        (each once per scan)."""
        hits = np.asarray(hits_xy, float).reshape(-1, 2)
        if not len(hits):
            return
        # One ray per hit cell, not per point: a lidar sweep has thousands of points, mostly
        # several to a cell. Cells are handled as flat indices (fast to make unique).
        hit_idx = self._flat(hits)
        if not len(hit_idx):
            return
        hits = self.to_xy(np.stack(np.unravel_index(hit_idx, (self.n, self.n)), axis=1))
        sensor = np.asarray(sensor_xy, float)
        d = np.linalg.norm(hits - sensor, axis=1)
        reach = np.minimum(d - self.res, free_range_m)          # free up to a cell short of the hit
        steps = np.arange(0, max(float(reach.max()), 0.0), self.res * 0.75)
        if len(steps):
            unit = (hits - sensor) / np.maximum(d[:, None], 1e-9)
            keep = steps[None, :] < reach[:, None]
            samples = sensor + steps[None, :, None] * unit[:, None, :]
            free = np.zeros(self.n * self.n, bool)
            free[self._flat(samples[keep])] = True
            free[hit_idx] = False                                # hit this scan: not also freed
            self.logodds.reshape(-1)[free] += L_FREE
        self.logodds.reshape(-1)[hit_idx] += L_HIT
        np.clip(self.logodds, L_MIN, L_MAX, out=self.logodds)
        # Grown obstacles: add the new hits to the cached maps (cheap), and rebuild them from
        # scratch every few scans so cells the rays have cleared open up again.
        self._updates += 1
        if self._updates % REGROW_EVERY == 0:
            self._blocked = {}
        else:
            now_occupied = hit_idx[self.logodds.reshape(-1)[hit_idx] > L_OCCUPIED]
            cells = np.stack(np.unravel_index(now_occupied, (self.n, self.n)), axis=1)
            for radius, grown in self._blocked.items():
                self._stamp(grown, cells, radius)

    def _flat(self, xy):
        """Unique flat indices of the cells at points xy (inside the map)."""
        cells = self.to_cell(xy).reshape(-1, 2)
        cells = cells[self.inside(cells)]
        return np.unique(cells[:, 0] * self.n + cells[:, 1]) if len(cells) else np.zeros(0, int)

    def mark(self, xy):
        """Mark the cells at points xy as surely occupied (e.g. something bumped into that the
        lidar doesn't see). Clearing them takes many scans seeing through them."""
        cells = self._unique_cells(np.asarray(xy, float).reshape(-1, 2))
        if len(cells):
            self.logodds[cells[:, 0], cells[:, 1]] = L_MAX
            self._blocked = {}

    def _unique_cells(self, xy):
        cells = self.to_cell(xy)
        cells = cells[self.inside(cells)]
        return np.unique(cells, axis=0) if len(cells) else cells

    def occupied(self):
        return (self.logodds > L_OCCUPIED) | self.steep

    def update_ground(self, xy, up, step_m, max_slope_deg):
        """Add surface points (xy, and height up in m, positive up) to the height map and mark
        cells where the ground rises more than step_m within 0.5 m (the top of a kerb, a ramp's
        edge), or more than max_slope_deg over 1 m (a bank, a ramp), as too steep to drive on."""
        idx = self.to_cell(np.asarray(xy, float).reshape(-1, 2))
        inside = self.inside(idx)
        if not inside.any():
            return
        idx, up = idx[inside], np.asarray(up, np.float32)[inside]
        flat = idx[:, 0] * self.n + idx[:, 1]
        g = self.ground.reshape(-1)
        np.fmin.at(g, flat, up)
        # Recheck only the window around this scan's cells.
        pad = int(math.ceil(1.0 / self.res)) + 1
        i0, j0 = np.maximum(idx.min(axis=0) - pad, 0)
        i1, j1 = np.minimum(idx.max(axis=0) + pad + 1, self.n)
        win = self.ground[i0:i1, j0:j1]
        steep = np.zeros(win.shape, bool)
        short, long_ = max(1, round(0.5 / self.res)), max(1, round(1.0 / self.res))
        for k, limit in ((short, step_m), (long_, long_ * self.res * math.tan(math.radians(max_slope_deg)))):
            for di, dj in ((k, 0), (0, k), (k, k), (k, -k)):
                limit_here = limit * (math.sqrt(2) if di and dj else 1.0)
                a = win[max(di, 0):win.shape[0] + min(di, 0) or None, max(dj, 0):win.shape[1] + min(dj, 0) or None]
                b = win[max(-di, 0):win.shape[0] + min(-di, 0) or None, max(-dj, 0):win.shape[1] + min(-dj, 0) or None]
                # Mark the higher cell of a steep pair (the ramp, the top of the step): marking both
                # would also block the flat ground at the foot of every wall and ramp.
                with np.errstate(invalid="ignore"):
                    a_up = (a - b) > limit_here            # nan (unseen) compares False
                    b_up = (b - a) > limit_here
                steep[max(di, 0):steep.shape[0] + min(di, 0) or None,
                      max(dj, 0):steep.shape[1] + min(dj, 0) or None] |= a_up
                steep[max(-di, 0):steep.shape[0] + min(-di, 0) or None,
                      max(-dj, 0):steep.shape[1] + min(-dj, 0) or None] |= b_up
        if (steep != self.steep[i0:i1, j0:j1]).any():
            self.steep[i0:i1, j0:j1] = steep
            self._blocked = {}

    def blocked(self, radius_m):
        """Occupied cells grown by radius_m: where the robot's centre can't go (a copy)."""
        if radius_m not in self._blocked:
            self._blocked[radius_m] = self._grow(radius_m)
        return self._blocked[radius_m].copy()

    def set_bounds(self, lo_xy, hi_xy):
        """Keep the robot's centre inside this rectangle: everything outside is blocked."""
        self.bounds = (np.asarray(lo_xy, float), np.asarray(hi_xy, float))
        self._blocked = {}

    def _grow(self, radius_m):
        """Occupied cells and every cell within radius_m of one (obstacles are sparse, so the
        disk is stamped around each occupied cell rather than shifting the whole map), and
        everything outside the bounds."""
        out = np.zeros((self.n, self.n), bool)
        if self.bounds is not None:
            (i0, j0), (i1, j1) = self.to_cell(self.bounds[0]), self.to_cell(self.bounds[1])
            out[:] = True
            out[max(i0, 0):max(i1 + 1, 0), max(j0, 0):max(j1 + 1, 0)] = False
        self._stamp(out, np.argwhere(self.occupied()), radius_m)
        return out

    def _stamp(self, out, cells, radius_m):
        """Mark every cell within radius_m of cells in out."""
        if not len(cells):
            return
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

    # ---- diagnosis ---------------------------------------------------------------------

    def goal_blocked(self, goal_xy, radius_m, tolerance_m):
        """Whether every cell within tolerance_m of the goal is blocked."""
        blocked = self.blocked(radius_m)
        gi, gj = self.to_cell(goal_xy)
        r = int(math.ceil(tolerance_m / self.res))
        i0, i1, j0, j1 = max(gi - r, 0), min(gi + r + 1, self.n), max(gj - r, 0), min(gj + r + 1, self.n)
        ii, jj = np.mgrid[i0:i1, j0:j1]
        near = (ii - gi) ** 2 + (jj - gj) ** 2 <= r * r
        return bool(blocked[i0:i1, j0:j1][near].all()) if near.any() else True

    def enclosed(self, start_xy, radius_m, window_m=15.0):
        """Whether the robot at start_xy is boxed in: no way out to the edge of a window_m square
        around it (blocked by obstacles grown by radius_m)."""
        blocked = self.blocked(radius_m)
        si, sj = self.to_cell(start_xy)
        r = int(math.ceil(radius_m / self.res))
        blocked[max(si - r, 0):si + r + 1, max(sj - r, 0):sj + r + 1] = False
        w = int(window_m / 2 / self.res)
        seen, todo = {(si, sj)}, [(si, sj)]
        while todo:
            i, j = todo.pop()
            if abs(i - si) >= w or abs(j - sj) >= w:
                return False
            for di, dj, _ in NEIGHBOURS:
                c = (i + di, j + dj)
                if c not in seen and 0 <= c[0] < self.n and 0 <= c[1] < self.n and not blocked[c]:
                    seen.add(c)
                    todo.append(c)
        return True

    def picture(self, radius_m, marks, margin_m=12.0, scale=4):
        """A BGR image of the map around the marks [(xy, (b, g, r))]: obstacles black, grown
        obstacles grey, seen-free white, unknown light grey. North (+x) is up, east (+y) right."""
        pts = self.to_cell([m[0] for m in marks])
        m = int(margin_m / self.res)
        i0, j0 = np.maximum(pts.min(axis=0) - m, 0)
        i1, j1 = np.minimum(pts.max(axis=0) + m + 1, self.n)
        img = np.full((i1 - i0, j1 - j0, 3), 200, np.uint8)
        img[self.logodds[i0:i1, j0:j1] < 0] = 255
        img[self.blocked(radius_m)[i0:i1, j0:j1]] = 140
        img[self.occupied()[i0:i1, j0:j1]] = 0
        for (xy, color) in marks:
            i, j = self.to_cell(xy)
            img[max(i - 2 - i0, 0):i + 3 - i0, max(j - 2 - j0, 0):j + 3 - j0] = color
        img = img[::-1]  # rows: north up
        return np.kron(img, np.ones((scale, scale, 1), np.uint8))
