"""Figure 1 (the overview) and Figure 2 (the modules) of the paper, as TikZ.

Both figures are drawn by code, so every number, cell and dot in them comes from the inputs of the export: the audit
of the corpus, its list of models and benchmarks, and the per-model values of the diagnostics.  The design is that of
two figures drawn by hand for an earlier draft; coordinates are in the units of those drawings (``WIDTH_1`` and
``WIDTH_2`` units wide, y pointing down) and are scaled to the column and to the text width.

Text is set by LaTeX in the paper's sans-serif face (Helvetica).  Its width is estimated here from the Adobe font
metrics that matplotlib ships, so that a text that would run into its neighbour or off the drawing stops the export.
"""

import math
import re
from pathlib import Path

WIDTH_1, WIDTH_2 = 818.88, 1700.88                 # units across Figure 1 (one column) and Figure 2 (text width)
SCALE_1 = 7.7 / 2.54 * 72.27 / WIDTH_1             # points per unit: a column of 7.7 cm
SCALE_2 = 16.0 / 2.54 * 72.27 / WIDTH_2            # the text width of 16 cm
BODY, TITLE, PANEL = 20.2, 21.0, 26.2              # text sizes in units (5.4, 5.6 and 7.0 pt when printed)
COLOURS = {"drink": "0B0B0B", "drgrey": "52514E", "drmid": "898781", "drstrip": "F1F0EC", "drline": "D5D4CC",
           "drrule": "C3C2B7", "drcell": "3987E5", "drmodel": "2A78D6", "drnavy": "104281", "drorange": "EB6834",
           "drtile": "F3F3F1", "drpale": "B7D3F6"}
AREAS = (("knowledge", "Knowledge"), ("language", "Language"), ("retrieval", "Retrieval"), ("tools", "Tools"),
         ("arts", "Arts"))
GLYPHS = {"·": "periodcentered", "×": "multiply", "’": "quoteright"}      # characters outside ASCII, by glyph name
TEX = {"$": r"\$", "%": r"\%", "&": r"\&", "#": r"\#", "_": r"\_", "·": r"\textperiodcentered{}", "×": r"\texttimes{}",
       "’": "'"}
_METRICS = {}


def metrics(bold):
    """Return the widths of Helvetica or Helvetica Bold in thousandths of the size.

    The keys are character codes (the font's own encoding agrees with ASCII for the characters used here) and, for
    the characters of ``GLYPHS``, glyph names.
    """
    name = "phvb8a.afm" if bold else "phvr8a.afm"
    if name not in _METRICS:
        import matplotlib

        table = {}
        text = (Path(matplotlib.get_data_path()) / "fonts" / "afm" / name).read_text(encoding="latin-1")
        for match in re.finditer(r"^C\s+(-?\d+)\s*;\s*WX\s+(\d+)\s*;\s*N\s+(\S+)", text, re.MULTILINE):
            table[match[3]] = int(match[2])
            if int(match[1]) >= 0:
                table[int(match[1])] = int(match[2])
        _METRICS[name] = table
    return _METRICS[name]


def glyph_width(table, character):
    """Return the width of one character, or stop: a character without a known width cannot be checked for fit."""
    key = GLYPHS.get(character, ord(character))
    if key not in table or character in "'`":           # the font encodes these two differently from ASCII
        raise ValueError(f"no width is known for the character {character!r} of a figure text")
    return table[key]


def parts_of(text):
    """Return [(text, bold)] for a string or a list of strings and (string, bold) pairs."""
    if isinstance(text, str):
        return [(text, False)]
    return [(part, False) if isinstance(part, str) else (part[0], bool(part[1])) for part in text]


def text_width(text, size):
    """Return the estimated width of a text in units (kerning is not counted, so this is an upper bound)."""
    total = 0.0
    for part, bold in parts_of(text):
        table = metrics(bold)
        total += sum(glyph_width(table, ch) for ch in part) * size / 1000
    return total


def tex(text):
    """Return a text as LaTeX, with bold parts marked."""
    out = []
    for part, bold in parts_of(text):
        escaped = "".join(TEX.get(ch, ch) for ch in part)
        out.append(r"\textbf{" + escaped + "}" if bold else escaped)
    return "".join(out)


def count(value):
    """Return an integer with thousands separators."""
    if value != int(value):
        raise ValueError("a count in a figure has to be an integer")
    return f"{int(value):,}"


class Canvas:
    """TikZ commands in the units of a drawing, with the extent of every text kept for the fit check."""

    def __init__(self, width, scale):
        self.width, self.scale, self.commands, self.texts = width, scale, [], []

    def pt(self, units):
        return f"{units * self.scale:.3f}pt"

    @staticmethod
    def at(x, y):
        return f"({x:.2f},{y:.2f})"

    def note(self, text):
        """Add a comment line; the tests find the grid and the dots by these."""
        self.commands.append("% " + text)

    def style(self, fill=None, stroke=None, width=None, radius=0.0, dash=False, cap=None):
        options = []
        if fill:
            options.append("fill=" + fill)
        if stroke:
            options += ["draw=" + stroke, "line width=" + self.pt(width)]
            if dash:
                options.append(f"dash pattern=on {self.pt(4.2)} off {self.pt(3.4)}")
            if cap:
                options += ["line cap=" + cap, "line join=" + cap]
        if radius:
            options.append("rounded corners=" + self.pt(radius))
        return ", ".join(options)

    def rect(self, x0, y0, x1, y1, **style):
        self.commands.append(rf"\path[{self.style(**style)}] {self.at(x0, y0)} rectangle {self.at(x1, y1)};")

    def rects(self, boxes, fill):
        """Fill many rectangles as one path."""
        path = " ".join(f"{self.at(x0, y0)} rectangle {self.at(x1, y1)}" for x0, y0, x1, y1 in boxes)
        self.commands.append(rf"\path[fill={fill}] {path};")

    def line(self, points, stroke, width, cap="round", dash=False, close=False, fill=None):
        path = " -- ".join(self.at(x, y) for x, y in points) + (" -- cycle" if close else "")
        self.commands.append(rf"\path[{self.style(fill=fill, stroke=stroke, width=width, dash=dash, cap=cap)}] {path};")

    def polygon(self, points, fill):
        path = " -- ".join(self.at(x, y) for x, y in points)
        self.commands.append(rf"\path[fill={fill}] {path} -- cycle;")

    def curve(self, p0, p1, p2, p3, stroke, width, dash=False):
        self.commands.append(rf"\path[{self.style(stroke=stroke, width=width, dash=dash, cap='round')}] "
                             rf"{self.at(*p0)} .. controls {self.at(*p1)} and {self.at(*p2)} .. {self.at(*p3)};")

    def circle(self, x, y, radius, **style):
        self.commands.append(rf"\path[{self.style(**style)}] {self.at(x, y)} circle[radius={self.pt(radius)}];")

    def text(self, x, y, text, size=BODY, colour="drink", align="left", rotate=False, name=None):
        """Put a line of text with its baseline at y; ``align`` says which point of it lies at x."""
        width = text_width(text, size)
        anchor = {"left": "base west", "centre": "base", "right": "base east"}[align]
        left = {"left": x, "centre": x - width / 2, "right": x - width}[align]
        font = rf"\sffamily\fontsize{{{size * self.scale:.2f}}}{{{size * self.scale:.2f}}}\selectfont"
        options = f"anchor={anchor}, inner sep=0pt, outer sep=0pt, text={colour}, font={font}"
        if rotate:
            options += ", rotate=90"
        self.commands.append(rf"\node[{options}] at {self.at(x, y)} {{{tex(text)}}};")
        plain = "".join(part for part, _ in parts_of(text))
        if rotate:
            self.texts.append(dict(name=name or plain, rotated=True, low=y - width / 2, high=y + width / 2, size=size))
        else:
            self.texts.append(dict(name=name or plain, rotated=False, left=left, right=left + width, base=y, size=size))
        return left, left + width

    def check_fit(self, margin=6.0):
        """Raise ValueError when a text leaves the drawing or runs into another text on its line."""
        flat = [t for t in self.texts if not t["rotated"]]
        for t in flat:
            if t["left"] < -0.5 or t["right"] > self.width + 0.5:
                raise ValueError(f"figure text does not fit the drawing: {t['name']}")
        for i, a in enumerate(flat):
            for b in flat[i + 1:]:
                same_line = abs(a["base"] - b["base"]) < 0.8 * min(a["size"], b["size"])
                if same_line and a["left"] < b["right"] + margin and b["left"] < a["right"] + margin:
                    raise ValueError(f"figure texts run into each other: {a['name']} / {b['name']}")

    def render(self, height):
        """Return the picture; its bounding box is the whole drawing."""
        self.check_fit()
        head = [rf"\definecolor{{{name}}}{{HTML}}{{{value}}}%" for name, value in COLOURS.items()]
        head.append(rf"\begin{{tikzpicture}}[x={self.scale:.6f}pt, y=-{self.scale:.6f}pt]")
        head.append(rf"\useasboundingbox (0,0) rectangle {self.at(self.width, height)};")
        return "\n".join(head + self.commands + [r"\end{tikzpicture}%"]) + "\n"


# Shapes that both figures use

def bars(canvas, x, bottom, heights, width, pitch, fill, radius=1.0):
    """Draw a small bar chart: the pictogram of a probability vector."""
    for i, height in enumerate(heights):
        canvas.rect(x + i * pitch, bottom - height, x + i * pitch + width, bottom, fill=fill, radius=radius)


def arrow_head(canvas, x, y, direction="right", size=8.4):
    """Draw a filled arrow head with its tip at (x, y)."""
    half = 0.55 * size
    if direction == "right":
        canvas.polygon([(x, y), (x - size, y - half), (x - size, y + half)], "drink")
    else:
        canvas.polygon([(x, y), (x - half, y - size), (x + half, y - size)], "drink")


def diamond(canvas, x, y, half):
    canvas.line([(x - half, y), (x, y - half), (x + half, y), (x, y + half)], "drink", 2.1, cap="butt", close=True,
                fill="white")


def square(canvas, x, y, size, fill="drmodel", dashed=False):
    """Draw a model as a rounded square centred at (x, y); an outlined one is a model that is not called."""
    half = size / 2
    if fill:
        canvas.rect(x - half, y - half, x + half, y + half, fill=fill, radius=3.4)
    else:
        canvas.rect(x - half + 1, y - half + 1, x + half - 1, y + half - 1, fill="white", stroke="drmid", width=2.1,
                    radius=3.0, dash=dashed)


# Figure 1

def axis(kind, values):
    """Return (ticks in axis units, transform, first label, last label) of a dot strip that holds the values."""
    if kind == "accuracy":
        shown = [100 * v for v in values]
        step = 10
        low, high = step * math.floor(min(shown) / step), step * math.ceil(max(shown) / step)
        if high - low > 4 * step:
            step = 20
            low, high = step * math.floor(low / step), step * math.ceil(high / step)
        high = max(high, low + step)
        ticks = list(range(int(low), int(high) + 1, step))
        return ticks, (lambda v: 100 * v), f"{ticks[0]}", f"{ticks[-1]}%"
    if kind == "cost":
        if min(values) <= 0:
            raise ValueError("a cost of zero cannot be drawn on a logarithmic axis")
        low, high = math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))
        high = max(high, low + 1)
        ticks = list(range(low, high + 1))

        def dollars(exponent):
            return "$" + (f"{10 ** exponent:,.0f}" if exponent >= 0 else f"{10 ** exponent:.{-exponent}f}")
        return ticks, math.log10, dollars(ticks[0]), dollars(ticks[-1])
    step = 0.5 if max(values) <= 2 else 1.0
    high = max(step * math.ceil(max(values) / step), step)
    ticks = [i * step for i in range(int(round(high / step)) + 1)]
    return ticks, float, "0", f"{high:g} s"


def grid_layout(models, benchmarks):
    """Return the boxes of the grid: rows are models grouped by organisation, columns benchmarks grouped by area.

    models: [(model index, organisation)] in the order of the corpus; benchmarks: [(catalog id, area)].
    Returns (columns, rows, column blocks, row blocks): the first two are [(low, high)] in units, the blocks the
    number of columns per task area (with its name) and of rows per organisation.
    """
    left, right, top, target = 39.0, 526.5, 87.75, 244.0
    gap, area_gap, organisation_gap = 1.5, 9.0, 6.0
    known = {area for area, _ in AREAS}
    if any(area not in known for _, area in benchmarks):
        raise ValueError("a benchmark lies outside the task areas of the overview")
    column_blocks = [(title, sum(a == area for _, a in benchmarks)) for area, title in AREAS]
    column_blocks = [(title, n) for title, n in column_blocks if n]
    row_blocks = []
    for _, organisation in models:
        if row_blocks and row_blocks[-1][0] == organisation:
            row_blocks[-1][1] += 1
        else:
            row_blocks.append([organisation, 1])
    if len({name for name, _ in row_blocks}) != len(row_blocks):
        raise ValueError("the models of an organisation are not listed together")
    n_columns, n_rows = len(benchmarks), len(models)
    gaps = (n_columns - len(column_blocks)) * gap + (len(column_blocks) - 1) * area_gap
    cell_width = (right - left - gaps) / n_columns
    gaps = (n_rows - len(row_blocks)) * gap + (len(row_blocks) - 1) * organisation_gap
    cell_height = min(cell_width, (target - gaps) / n_rows)
    columns, x = [], left
    for _, n in column_blocks:
        for _ in range(n):
            columns.append((x, x + cell_width))
            x += cell_width + gap
        x += area_gap - gap
    rows, y = [], top
    for _, n in row_blocks:
        for _ in range(n):
            rows.append((y, y + cell_height))
            y += cell_height + gap
        y += organisation_gap - gap
    return columns, rows, column_blocks, [n for _, n in row_blocks]


STRIPS = (       # key of the per-model value, title, note under the axis, left edge, first and last grid line
    ("accuracy", "accuracy", "mean", 543.0, 553.5, 606.0),
    ("cost", "cost", "per 1k req.", 639.0, 663.0, 711.0),
    ("latency", "latency", "median", 735.0, 742.5, 795.0),
)


def overview(models, benchmarks, values, totals, released=True):
    """Return the overview as TikZ.

    models: [(model index, organisation)] in the order of the corpus; benchmarks: [(catalog id, area)];
    values: {model index: {"accuracy": share, "cost": USD per 1,000 requests, "latency": seconds}};
    totals: requests, fields, vectors, organisations of the corpus; released: the whole corpus is released.
    The corpus calls the party that offers a model its organisation; the paper, and so the picture, its provider.
    """
    if sorted(values) != sorted(index for index, _ in models):
        raise ValueError("the per-model values do not cover the models of the corpus")
    c = Canvas(WIDTH_1, SCALE_1)
    columns, rows, column_blocks, row_blocks = grid_layout(models, benchmarks)
    bottom, delta = rows[-1][1], rows[-1][1] - 324.75        # everything under the grid moves with its height
    n_models, n_benchmarks = len(models), len(benchmarks)

    # (a) the corpus
    c.text(0, 25.5, [("a", True)], PANEL)
    c.text(25.5, 23.2, [("Corpus: every model answered every request", True)], TITLE)
    c.text(39, 52.5, [(count(n_benchmarks), True), " fixed-choice benchmarks in five task areas"], colour="drgrey")
    start = 0
    for title, n in column_blocks:
        low, high = columns[start][0], columns[start + n - 1][1]
        c.rect(low, 82.5, high, 84.0, fill="drrule")
        c.text((low + high) / 2, 75.8, title, align="centre")
        start += n
    c.note(f"cells: {n_models} models x {n_benchmarks} benchmarks")
    c.rects([(x0, y0, x1, y1) for y0, y1 in rows for x0, x1 in columns], "drcell")
    beside = [(count(n_models), True), " models, ", (count(totals["organisations"]), True), " providers"]
    c.text(24.0, (rows[0][0] + bottom) / 2, beside, colour="drgrey", align="centre", rotate=True)
    if text_width(beside, BODY) > bottom - 52.5:
        raise ValueError("the label of the rows is longer than the room beside the grid")

    c.text(543, 52.5, "per model", colour="drgrey")
    for key, title, note, left, first, last in STRIPS:
        right = left + 84.0
        ticks, transform, low_label, high_label = axis(key, [values[index][key] for index, _ in models])
        c.rect(left, 82.5, right, 84.0, fill="drrule")
        c.text(left + 42.0, 75.8, [(title, True)], align="centre")
        c.rects([(left, y0, right, y1) for y0, y1 in rows], "drstrip")
        step = (last - first) / (len(ticks) - 1)
        for i in range(len(ticks)):
            c.line([(first + i * step, 84.75), (first + i * step, bottom + 3.0)], "drline", 1.5, cap="butt")
        c.note(f"dots: {key}, axis {ticks[0]:g} to {ticks[-1]:g} from x={first:.2f} to x={last:.2f}")
        for (index, _), (y0, y1) in zip(models, rows):
            x = first + (transform(values[index][key]) - ticks[0]) / (ticks[-1] - ticks[0]) * (last - first)
            if not left + 3.0 <= x <= right - 3.0:
                raise ValueError(f"the {key} dot of model {index} lies outside its strip")
            c.circle(x, (y0 + y1) / 2, 3.0, fill="drgrey")
        c.text(first, 350.2 + delta, low_label, colour="drgrey", align="centre")
        c.text(last, 350.2 + delta, high_label, colour="drgrey", align="centre")
        c.text(left + 42.0, 372.8 + delta, note, colour="drgrey", align="centre")

    marked_row = rows[len(rows) // 2]
    marked_column = columns[sum(n for _, n in column_blocks[:len(column_blocks) // 2])
                            + min(2, column_blocks[len(column_blocks) // 2][1] - 1)]
    x0, x1, y0, y1 = *marked_column, *marked_row
    c.rect(x0 - 3.0, y0 - 3.0, x1 + 3.0, y1 + 3.0, fill="white", radius=3.0)
    c.rect(x0, y0, x1, y1, fill="drnavy")
    c.rect(x0 - 2.6, y0 - 2.6, x1 + 2.6, y1 + 2.6, stroke="drorange", width=2.25, radius=3.0)

    side = 9.0
    c.rect(39, 343.6 + delta - side / 2, 39 + side, 343.6 + delta + side / 2, fill="drcell")
    c.text(57, 350.2 + delta, ["all released: " if released else "", (count(n_models), True), " models × ",
                               (count(n_benchmarks), True), " benchmarks"], colour="drgrey")
    c.text(57, 372.8 + delta, [(count(totals["requests"]), True), " requests · ", (count(totals["fields"]), True),
                               " questions"], colour="drgrey")

    # (b) the record
    y = delta
    c.text(0, 418.5 + y, [("b", True)], PANEL)
    c.rect(31.5, 403.0 + y, 43.5, 415.0 + y, fill="drnavy")
    c.rect(28.9, 400.4 + y, 46.1, 417.6 + y, stroke="drorange", width=2.25, radius=3.0)
    c.text(55.5, 416.2 + y, [("Each cell holds one record per request", True)], TITLE)
    c.text(WIDTH_1, 415.5 + y, [(count(totals["vectors"]), True), " probability vectors"], align="right")
    c.line([(25.5, 471.0 + y), (67.5, 471.0 + y)], "drmid", 1.5)
    for i, height in enumerate((7.4, 15.0, 33.0, 11.2, 5.2)):
        c.rect(27.8 + 7.85 * i, 470.2 + y - height, 33.8 + 7.85 * i, 470.2 + y, fill="drnavy" if i == 2 else "drcell",
               radius=1.0)
    c.text(78, 450.8 + y, [("Probability vector", True)], TITLE)
    c.text(78, 472.5 + y, "for every answer option", colour="drgrey")
    c.circle(376.75, 453.0 + y, 18.4, stroke="drgrey", width=1.875)
    c.text(376.75, 462.0 + y, [("$", True)], 24.8, colour="drgrey", align="centre", name="dollar sign")
    c.text(408.2, 450.8 + y, [("Billed cost", True)], TITLE)
    c.text(408.2, 472.5 + y, "as charged", colour="drgrey")
    c.circle(603.0, 453.0 + y, 18.4, stroke="drgrey", width=1.875)
    c.line([(603.0, 441.8 + y), (603.0, 453.0 + y), (610.9, 457.9 + y)], "drgrey", 2.25)
    c.text(634.5, 450.8 + y, [("Measured latency", True)], TITLE)
    c.text(634.5, 472.5 + y, "as timed", colour="drgrey")
    c.text(25.5, 501.8 + y, "Also stored: reported choice · token counts · number of calls · fold and group",
           colour="drgrey")

    # (c) the simulator and the three families of policies
    c.text(0, 547.5 + y, [("c", True)], PANEL)
    c.text(25.5, 545.2 + y, [("Simulator: replays and scores any policy, with no new model call", True)], TITLE)
    middle = 605.6 + y
    lanes = [572.8 + y, 594.7 + y, 616.5 + y, 638.4 + y]
    vectors = ((5.9, 13.4, 8.4), (12.5, 6.7, 9.2), (5.1, 9.3, 14.3), (7.5, 14.2, 5.9))

    c.circle(33.9, middle, 5.9, fill="drorange")                               # combine
    for lane, heights in zip(lanes, vectors):
        c.curve((39.8, middle), (62.5, middle), (59.1, lane), (80.9, lane), "drorange", 2.1)
        square(c, 91.05, lane, 18.5)
        bars(c, 107.8, lane + 7.6, heights, 4.2, 6.3, "drcell")
        c.curve((129.7, lane), (161.6, lane), (158.2, middle), (189.3, middle), "drcell", 1.68)
    c.circle(200.2, middle, 10.5, fill="white", stroke="drink", width=2.1)
    c.line([(195.2, middle), (205.3, middle)], "drink", 2.1)
    c.line([(200.2, middle - 5.0), (200.2, middle + 5.1)], "drink", 2.1)
    c.line([(211.1, middle), (230.5, middle)], "drink", 2.1)
    arrow_head(c, 238.9, middle)
    bars(c, 248.9, middle + 16.8, (12.6, 30.2, 18.5), 8.4, 11.8, "drnavy")
    c.line([(244.7, middle + 18.5), (285.1, middle + 18.5)], "drmid", 1.68)

    c.circle(342.9, middle, 5.9, fill="drorange")                              # route
    c.line([(348.8, middle), (368.1, middle)], "drorange", 2.1)
    for i, lane in enumerate(lanes):
        chosen = i == 1
        c.curve((391.6, middle), (411.8, middle), (411.8, lane), (431.9, lane), "drorange" if chosen else "drmid",
                2.1 if chosen else 1.68, dash=not chosen)
        square(c, 442.05, lane, 18.5, fill="drmodel" if chosen else None)
    diamond(c, 379.85, middle, 10.95)
    bars(c, 458.8, lanes[1] + 7.5, (5.8, 13.4, 8.4), 4.2, 6.3, "drcell")
    c.curve((480.7, lanes[1]), (500.8, lanes[1]), (499.1, middle), (515.9, middle), "drink", 2.1)
    arrow_head(c, 524.3, middle)
    bars(c, 534.4, middle + 16.8, (12.6, 30.2, 18.5), 8.4, 11.75, "drnavy")
    c.line([(530.2, middle + 18.5), (570.5, middle + 18.5)], "drmid", 1.68)

    track = 587.1 + y                                                          # cascade
    c.circle(594.15, track, 5.9, fill="drorange")
    for low, high in ((600.0, 624.4), (644.5, 661.3), (678.2, 696.6), (720.2, 740.3)):
        c.line([(low, track), (high, track)], "drorange", 2.1)
    square(c, 634.5, track, 16.8)
    diamond(c, 669.75, track, 7.95)
    square(c, 708.4, track, 20.2)
    diamond(c, 748.7, track, 7.95)
    c.line([(756.3, track), (779.8, track)], "drmid", 1.68, dash=True)
    square(c, 794.05, track, 25.1, fill=None, dashed=True)
    c.line([(748.7, track + 8.4), (748.7, track + 21.9)], "drink", 2.1)
    arrow_head(c, 748.7, track + 30.3, direction="down")
    c.text(759, 617.2 + y, "stop", colour="drgrey")
    bars(c, 732.8, 647.6 + y, (12.6, 25.2, 18.5), 8.4, 11.75, "drnavy")
    c.line([(728.5, 649.3 + y), (768.9, 649.3 + y)], "drmid", 1.68)

    for x, title, line, methods in (
            (25.5, "Combine", "pool several models’ vectors",
             ("Equal-weight pool · Majority vote", "Skill-weighted pool · Stacking", "Greedy selection")),
            (334.5, "Route", "one model per request", ("kNN router · Ridge router",)),
            (585.8, "Cascade", "call in stages, stop early", ("Confidence cascade", "Certified early exit"))):
        c.text(x, 676.5 + y, [(title, True)], TITLE)
        c.text(x, 699.0 + y, line)
        for i, method in enumerate(methods):
            c.text(x, 726.0 + y + 22.5 * i, method, colour="drgrey")
    return c.render(776.25 + delta)


# Figure 2

def modules(totals):
    """Return the strip of the five modules as TikZ.  totals: models, observations and cost_usd of the corpus."""
    c = Canvas(WIDTH_2, SCALE_2)
    pitch = 345.75

    for low, high, text in (
            (0.0, 663.75, [("Collected once", True),
                           f" · all model calls happen here, ${totals['cost_usd']:,.2f} in total"]),
            (691.5, WIDTH_2, [("Replayed offline", True), " · any policy from the corpus, with no new model call"])):
        c.rect(low, 14.25, high, 16.5, fill="drgrey")
        c.rect(low, 14.25, low + 2.25, 24.0, fill="drgrey")
        c.rect(high - 2.25, 14.25, high, 24.0, fill="drgrey")
        half = text_width(text, BODY) / 2 + 13.0
        c.rect((low + high) / 2 - half, 6.0, (low + high) / 2 + half, 26.0, fill="white")
        c.text((low + high) / 2, 21.0, text, align="centre")

    lines = (
        ("Collector", ("Identical requests, one gateway;",
                       [(count(totals["observations"]), True), " records with tokens,"],
                       "billed cost and measured latency.")),
        ("Corpus builder", ("Vector, usage and latency tables", "with hashed ids; labels rebuilt",
                            "from each benchmark’s release.")),
        ("Simulator", ("Replays call plans in stages: cost", "is the sum, latency the slowest",
                       "call per stage. Five-fold rotation.")),
        ("Policies", ("One interface for combination,", "routing and cascades: which models",
                      "to call, how to pool, when to stop.")),
        ("Metrics", ("Scores per judged question,", "equal weight per benchmark,", "paired bootstrap intervals.")))
    for i, (title, text) in enumerate(lines):
        x = i * pitch
        _, right = c.text(x, 60.8, [(str(i + 1), True)], PANEL)
        c.text(right + 7.5, 58.5, [(title, True)], TITLE)
        for j, line in enumerate(text):
            c.text(x, 200.2 + 24.0 * j, line)
        if i:
            c.line([(x - 24.0, 111.75), (x - 8.2, 111.75)], "drink", 1.88, cap="butt")
            arrow_head(c, x - 4.5, 111.75, size=7.5)

    def label(x, y, text):
        c.text(x, y, text, colour="drgrey", align="centre")

    # 1 collector: a request goes through one gateway to every model, which returns a vector, a cost and a latency
    c.rect(2.2, 79.5, 51.8, 144.0, fill="white", stroke="drorange", width=1.88, radius=6.0)
    c.line([(10.5, 91.5), (43.5, 91.5)], "drgrey", 2.25, cap="butt")
    c.line([(10.5, 99.8), (33.0, 99.8)], "drgrey", 2.25, cap="butt")
    for j, yy in enumerate((114.0, 124.5, 135.0)):
        c.circle(12.7, yy, 2.7, fill="drgrey" if j == 1 else "white", stroke="drgrey", width=1.5)
        c.line([(20.2, yy), (42.0, yy)], "drmid", 1.88, cap="butt")
    c.line([(52.5, 111.75), (111.0, 111.75)], "drorange", 1.88, cap="butt")
    c.rect(111.0, 81.0, 123.0, 142.5, fill="drink", radius=6.0)
    for lane, heights in zip((84.8, 102.8, 120.8, 138.8),
                             ((5.3, 12.0, 7.5), (10.5, 6.0, 8.3), (4.5, 8.3, 12.7), (6.7, 12.7, 5.3))):
        c.curve((123.8, 111.75), (154.5, 111.75), (154.5, lane), (184.5, lane), "drorange", 1.88)
        square(c, 193.5, lane - 0.1, 15.0)
        bars(c, 210.0, lane + 6.7, heights, 3.8, 5.6, "drcell")
        c.rect(231.0, lane - 3.0, 237.8, lane + 3.7, fill="drgrey", radius=1.0)
        c.rect(241.5, lane - 3.0, 248.2, lane + 3.7, fill="drmid", radius=1.0)
    c.text(1.5, 166.5, "request", colour="drgrey")
    label(117.0, 166.5, "gateway")
    label(218.6, 166.5, f"{count(totals['models'])} models")

    # 2 corpus builder: three tables of the corpus; the labels are not in it and are rebuilt by its user
    for j, name in enumerate(("vectors", "usage", "latency", "labels")):
        left = pitch + 2.25 + 72.0 * j
        inside = name != "labels"
        c.rect(left, 81.0, left + 63.0, 141.0, fill="white", stroke="drmodel" if inside else "drmid", width=1.88,
               radius=3.0, dash=not inside)
        if inside:
            c.rect(left, 81.0, left + 63.0, 94.5, fill="drmodel", radius=3.0)
            c.rect(left, 88.0, left + 63.0, 94.5, fill="drmodel")
        else:
            c.line([(left, 94.5), (left + 63.0, 94.5)], "drmid", 1.5, cap="butt", dash=True)
        for points in (((left, 110.25), (left + 63.0, 110.25)), ((left, 125.25), (left + 63.0, 125.25)),
                       ((left + 20.25, 94.5), (left + 20.25, 141.0))):
            c.line(points, "drpale" if inside else "drline", 1.5, cap="butt")
        label(left + 31.5, 162.0, name)

    # 3 simulator: calls of a stage run side by side; cost adds up, latency takes the slowest call of each stage
    c.text(714.0, 90.0, "stage 1", colour="drgrey")
    c.text(804.0, 90.0, "stage 2", colour="drgrey")
    c.rect(694.5, 99.0, 793.5, 112.5, fill="drnavy", radius=2.0)
    c.rect(694.5, 117.0, 760.5, 130.5, fill="drcell", radius=2.0)
    c.rect(808.5, 135.0, 864.0, 148.5, fill="drnavy", radius=2.0)
    c.line([(801.0, 93.8), (801.0, 152.2)], "drmid", 1.5, cap="butt", dash=True)
    c.line([(694.5, 157.5), (871.5, 157.5)], "drgrey", 1.88, cap="butt")
    c.polygon([(879.0, 157.5), (871.5, 153.4), (871.5, 161.6)], "drgrey")
    for top, fill in ((93.0, "drnavy"), (111.75, "drcell"), (130.5, "drnavy")):
        c.rect(903.75, top, 929.25, top + 16.5, fill=fill, radius=2.0)
    c.rect(962.25, 102.0, 987.75, 128.25, fill="drnavy", radius=2.0)
    c.rect(962.25, 130.5, 987.75, 147.0, fill="drnavy", radius=2.0)
    label(916.5, 166.5, "cost")
    label(975.0, 166.5, "latency")

    # 4 policies: which models to call, how to pool their vectors, when to stop
    for left in (1039.5, 1149.0, 1258.5):
        c.rect(left, 76.5, left + 94.5, 145.5, fill="drtile", radius=6.0)
    for column, row, called in ((0, 0, True), (1, 0, True), (2, 0, False), (0, 1, False), (1, 1, True), (2, 1, False)):
        square(c, 1062.75 + 24.0 * column, 99.0 + 24.0 * row, 16.5, fill="drmodel" if called else None)
    for lane, heights in zip((90.0, 111.0, 132.0), ((5.3, 12.0, 7.6), (10.6, 6.0, 8.3), (4.6, 8.3, 12.8))):
        bars(c, 1158.75, lane + 6.8, heights, 3.75, 5.6, "drcell")
        c.curve((1178.2, lane), (1191.8, lane), (1190.2, 111.0), (1202.2, 111.0), "drcell", 1.5)
    c.line([(1202.2, 111.0), (1208.2, 111.0)], "drink", 1.88, cap="butt")
    arrow_head(c, 1214.2, 111.0, size=7.4)
    bars(c, 1217.25, 123.0, (9.0, 22.5, 13.5), 6.0, 8.65, "drnavy")
    square(c, 1278.75, 111.0, 16.5)
    c.line([(1287.8, 111.0), (1293.8, 111.0)], "drorange", 1.88, cap="butt")
    square(c, 1302.75, 111.0, 16.5)
    c.line([(1316.25, 96.0), (1316.25, 126.0)], "drink", 2.25, cap="butt")
    square(c, 1332.75, 111.0, 16.5, fill=None, dashed=True)
    label(1086.75, 166.5, "call")
    label(1196.25, 166.5, "pool")
    label(1305.75, 166.5, "stop")

    # 5 metrics: a difference with its interval, and accuracy against cost
    c.line([(1389.0, 75.0), (1389.0, 145.5)], "drrule", 1.5, cap="butt")
    c.line([(1389.0, 109.5), (1529.4, 109.5)], "drgrey", 1.5, cap="butt", dash=True)
    for x, low, high, centre in ((1416.0, 120.8, 137.2, 129.0), (1449.0, 93.8, 107.2, 100.5),
                                 (1482.0, 94.5, 112.5, 103.5), (1515.0, 111.0, 126.0, 118.5)):
        c.line([(x, low), (x, high)], "drink", 1.88, cap="butt")
        c.circle(x, centre, 4.9, fill="drorange")
    c.line([(1557.0, 75.0), (1557.0, 145.5), (1698.0, 145.5)], "drrule", 1.5, cap="butt")
    c.curve((1563.0, 139.5), (1579.5, 103.5), (1621.5, 90.0), (1693.5, 84.0), "drmodel", 2.25)
    for x, yy in ((1579.5, 133.5), (1597.5, 123.0), (1608.0, 136.5), (1632.0, 115.5), (1647.0, 126.0),
                  (1671.0, 109.5), (1681.5, 123.0)):
        c.circle(x, yy, 3.75, fill="drmid")
    c.circle(1641.0, 102.0, 4.9, fill="drorange")
    label(1459.2, 166.5, "gain and interval")
    c.text(WIDTH_2 - 0.3, 166.5, "accuracy vs cost", colour="drgrey", align="right")
    return c.render(255.0)
