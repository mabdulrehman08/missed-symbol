"""Simulate drawing review with an OpenCV template matcher."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


SIZE = 25
HALF = SIZE // 2
ANGLES = (-90, -60, -30, 0, 30, 60, 90)
TUNE_SEEDS = range(1000, 1048)
TEST_SEEDS = range(2000, 2120)
GRID = np.arange(35, 96, 2) / 100
FP_BUDGET = 2
REVIEW_BUDGET = 6
IMAGE_SEED = 2000
BOOTSTRAP_SEED = 2026


def template(angle=0, variant=0):
    """An outlet-like symbol, with related service marks as distractors."""
    tile = np.full((SIZE, SIZE), 255, np.uint8)
    cv2.circle(tile, (12, 12), 7, 28, 1, cv2.LINE_AA)
    if variant != 1:
        cv2.line(tile, (10, 7), (10, 14), 25, 1, cv2.LINE_AA)
    if variant != 2:
        cv2.line(tile, (14, 7), (14, 14), 25, 1, cv2.LINE_AA)
    if variant == 3:
        cv2.line(tile, (6, 11), (18, 11), 25, 1, cv2.LINE_AA)
    cv2.line(tile, (12, 19), (12, 22), 25, 1, cv2.LINE_AA)
    transform = cv2.getRotationMatrix2D((HALF, HALF), angle, 1)
    return cv2.warpAffine(tile, transform, (SIZE, SIZE), borderValue=255)


def drawing(seed):
    rng = np.random.default_rng(seed)
    width, height = 660, 480
    sheet = np.full((height, width), 255, np.uint8)
    xs = [28, 174 + int(rng.integers(-15, 16)), 332, 499, width - 28]
    ys = [30, 172, 315 + int(rng.integers(-15, 16)), height - 28]

    # Room walls with openings, room names, and door swings.
    cv2.rectangle(sheet, (xs[0], ys[0]), (xs[-1], ys[-1]), 65, 3, cv2.LINE_AA)
    for y in ys[1:-1]:
        cv2.line(sheet, (xs[0], y), (xs[-1], y), 70, 3, cv2.LINE_AA)
    for x in xs[1:-1]:
        cv2.line(sheet, (x, ys[0]), (x, ys[-1]), 70, 3, cv2.LINE_AA)
    for row, y in enumerate(ys[:-1]):
        for col, x in enumerate(xs[:-1]):
            cv2.putText(sheet, f"R{row + 1}{col + 1}", (x + 9, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, .39, 107, 1, cv2.LINE_AA)
            if row:
                door = x + int(rng.integers(53, 97))
                cv2.line(sheet, (door, y), (door + 29, y), 255, 5)
                cv2.line(sheet, (door, y), (door, y + 29), 105, 1, cv2.LINE_AA)
                cv2.ellipse(sheet, (door, y), (29, 29), 0, 0, 90, 105, 1, cv2.LINE_AA)
            cabinet = (x + 14, ys[row + 1] - 33)
            cv2.rectangle(sheet, cabinet, (cabinet[0] + 35, cabinet[1] + 17), 120, 1)

    for _ in range(int(rng.integers(12, 29))):
        x, y = map(int, rng.integers([31, 31], [width - 95, height - 50]))
        length = int(rng.integers(12, 63))
        cv2.line(sheet, (x, y), (x + length, y), 147, 1, cv2.LINE_AA)
        cv2.line(sheet, (x, y - 4), (x, y + 4), 147, 1, cv2.LINE_AA)
        cv2.line(sheet, (x + length, y - 4), (x + length, y + 4), 147, 1, cv2.LINE_AA)

    count = int(rng.integers(13, 25))
    clutter = int(rng.integers(10, 20))
    locations = []
    while len(locations) < count + clutter:
        x, y = map(int, rng.integers([50, 52], [width - 50, height - 46]))
        if all((x - a) ** 2 + (y - b) ** 2 >= 34 ** 2 for a, b in locations):
            locations.append((x, y))
    for i, (x, y) in enumerate(locations):
        angle = rng.choice(ANGLES, p=[.08, .07, .14, .42, .14, .07, .08])
        angle = float(angle + rng.uniform(-12, 12))
        variant = 0 if i < count else int(rng.integers(1, 4))
        ink = template(angle, variant)
        faded = (255 - (255 - ink) * rng.uniform(.48, 1)).astype(np.uint8)
        patch = sheet[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1]
        np.minimum(patch, faded, out=patch)

    sigma = rng.uniform(.25, 1.15)
    blurred = cv2.GaussianBlur(sheet, (5, 5), sigma).astype(np.float32)
    gradient = np.linspace(float(rng.uniform(-27, -5)), 0, width)
    noisy = blurred + gradient + rng.normal(0, rng.uniform(4, 15), sheet.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8), locations[:count]


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    score: float


def suppress(candidates, existing=()):
    kept = list(existing)
    start = len(kept)
    for box in sorted(candidates, key=lambda b: (-b.score, b.y, b.x)):
        if all((box.x - b.x) ** 2 + (box.y - b.y) ** 2 > 16 ** 2 for b in kept):
            kept.append(box)
    return kept[start:]


def proposals(image, angles):
    scores = np.maximum.reduce([
        cv2.matchTemplate(image, template(angle), cv2.TM_CCOEFF_NORMED)
        for angle in angles
    ])
    local_max = cv2.dilate(scores, np.ones((5, 5), np.uint8))
    ys, xs = np.where((scores >= GRID[0]) & (scores == local_max))
    return [Box(int(x) + HALF, int(y) + HALF, float(scores[y, x]))
            for y, x in zip(ys, xs)]


def match(boxes, truth, available=None):
    """Assign score-ordered boxes to distinct truth centers within eight pixels."""
    remaining = set(range(len(truth))) if available is None else set(available)
    hits, wrong = {}, []
    for box in boxes:
        neighbors = [
            ((box.x - truth[i][0]) ** 2 + (box.y - truth[i][1]) ** 2, i)
            for i in remaining
        ]
        distance, i = min(neighbors, default=(float("inf"), -1))
        if distance <= 8 ** 2:
            hits[i] = box
            remaining.remove(i)
        else:
            wrong.append(box)
    return hits, wrong, remaining


def invisible_misses(boxes, truth, missed):
    """A miss is invisible if no emitted baseline box overlaps its truth box."""
    return {
        i for i in missed
        if not any(abs(b.x - truth[i][0]) < SIZE and abs(b.y - truth[i][1]) < SIZE
                   for b in boxes)
    }


@dataclass
class Sheet:
    seed: int
    truth: list
    upright: list
    rotated: list


def prepare(seeds):
    sheets = []
    for seed in seeds:
        image, truth = drawing(seed)
        sheets.append(Sheet(seed, truth, proposals(image, (0,)),
                            proposals(image, ANGLES)))
    return sheets


def above(boxes, threshold):
    return [b for b in boxes if b.score >= threshold]


def calibrate_baseline(sheets):
    options = []
    for threshold in GRID:
        total_hits, total_wrong = 0, 0
        for sheet in sheets:
            boxes = suppress(above(sheet.upright, threshold))
            hits, wrong, _ = match(boxes, sheet.truth)
            total_hits += len(hits)
            total_wrong += len(wrong)
        if total_wrong <= FP_BUDGET * len(sheets):
            options.append((total_hits, -total_wrong, threshold))
    return max(options)[2]


@dataclass
class Baseline:
    boxes: list
    hits: dict
    wrong: list
    missed: set
    invisible: set


def baseline(sheet, threshold):
    boxes = suppress(above(sheet.upright, threshold))
    hits, wrong, missed = match(boxes, sheet.truth)
    return Baseline(boxes, hits, wrong, missed,
                    invisible_misses(boxes, sheet.truth, missed))


def second_pass(sheet, base, threshold, source):
    boxes = suppress(above(getattr(sheet, source), threshold), base.boxes)
    hits, wrong, _ = match(boxes, sheet.truth, base.missed)
    return boxes, hits, wrong


def calibrate_review(sheets, baselines, source, baseline_threshold):
    options = []
    for threshold in GRID:
        if source == "upright" and threshold > baseline_threshold:
            continue
        total_added, total_recovered = 0, 0
        for sheet, base in zip(sheets, baselines):
            boxes, hits, _ = second_pass(sheet, base, threshold, source)
            total_added += len(boxes)
            total_recovered += len(base.invisible & hits.keys())
        if total_added <= REVIEW_BUDGET * len(sheets):
            options.append((total_recovered, -total_added, threshold))
    return max(options)[2]


def collect(sheets, baselines, thresholds):
    rows = []
    for sheet, base in zip(sheets, baselines):
        row = [len(sheet.truth), len(base.hits), len(base.wrong),
               len(base.missed), len(base.invisible), 1]
        for source, threshold in thresholds.items():
            boxes, hits, wrong = second_pass(sheet, base, threshold, source)
            row.extend([len(boxes), len(hits), len(base.invisible & hits.keys()), len(wrong)])
        rows.append(row)
    return np.array(rows, dtype=np.int64)


def ratio(totals, numerator, denominator):
    top, bottom = totals[..., numerator], totals[..., denominator]
    return np.divide(top, bottom, out=np.full(np.shape(top), np.nan), where=bottom != 0)


def metrics(totals):
    wrong, missed, invisible, sheets = np.moveaxis(totals[..., 2:6], -1, 0)
    yield "Baseline recall", 100 * ratio(totals, 1, 0), "%"
    yield "Baseline wrong boxes/sheet", wrong / sheets, ""
    error_counts = np.stack((invisible, missed - invisible, wrong, missed + wrong), axis=-1)
    for i, label in enumerate(("Invisible misses", "Misses with overlap", "Wrong boxes")):
        yield f"{label} / all baseline errors", 100 * ratio(error_counts, i, 3), "%"
    yield "Invisible misses / baseline misses", 100 * ratio(totals, 4, 3), "%"
    for label, start in (("Lower threshold", 6), ("Rotation bank", 10)):
        yield f"{label}: invisible misses recovered", 100 * ratio(totals, start + 2, 4), "%"
        yield f"{label}: extra boxes/sheet", totals[..., start] / sheets, ""
        yield f"{label}: extra wrong boxes/sheet", totals[..., start + 3] / sheets, ""
    yield "Rotation minus lower: recovery (percentage points)", (
        100 * (ratio(totals, 12, 4) - ratio(totals, 8, 4))
    ), ""
    yield "Rotation minus lower: extra boxes/sheet", (
        (totals[..., 10] - totals[..., 6]) / sheets
    ), ""


def print_counts(totals):
    truth, hits, wrong, missed, invisible = totals[:5]
    print(f"Baseline: {truth} symbols, {hits} hits, {wrong} wrong boxes, "
          f"{missed} misses ({invisible} invisible)")
    for label, start in (("Lower threshold", 6), ("Rotation bank", 10)):
        added, recovered, invisible, wrong = totals[start:start + 4]
        print(f"{label}: {added} added boxes, {recovered} recovered "
              f"({invisible} invisible), {wrong} wrong boxes")


def print_report(rows):
    totals = rows.sum(axis=0)
    print_counts(totals)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(len(rows), size=(10_000, len(rows)))
    boot_totals = rows[indices].sum(axis=1)
    for (label, value, unit), (_, samples, _) in zip(metrics(totals), metrics(boot_totals)):
        defined = samples[np.isfinite(samples)]
        if not np.isfinite(value) or not defined.size:
            print(f"{label}: undefined (zero denominator)")
            continue
        low, high = np.percentile(defined, [2.5, 97.5])
        print(f"{label}: {value:.2f}{unit} (95% CI {low:.2f} to {high:.2f}{unit})")


def render_example(path, sheet, base, rotation_threshold):
    image, _ = drawing(sheet.seed)
    color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = cv2.copyMakeBorder(color, 47, 79, 0, 0, cv2.BORDER_CONSTANT,
                                value=(249, 247, 241))
    _, hits, _ = second_pass(sheet, base, rotation_threshold, "rotated")
    blue, red, green, purple = (177, 96, 0), (45, 54, 204), (37, 129, 0), (147, 43, 125)
    ink = (46, 41, 30)

    def outline(x, y, box_color, dashed=False):
        left, right = x - HALF, x + HALF
        top, bottom = y + 47 - HALF, y + 47 + HALF
        if not dashed:
            cv2.rectangle(canvas, (left, top), (right, bottom), box_color, 2, cv2.LINE_AA)
            return
        for offset in range(0, SIZE - 1, 8):
            for t in (top, bottom):
                cv2.line(canvas, (left + offset, t), (left + offset + 4, t), box_color, 2)
            for side in (left, right):
                cv2.line(canvas, (side, top + offset), (side, top + offset + 4), box_color, 2)

    cv2.putText(canvas, f"Synthetic floor plan  /  held-out sheet {sheet.seed}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .69, ink, 1, cv2.LINE_AA)
    for box in base.hits.values():
        outline(box.x, box.y, blue)
    for box in base.wrong:
        outline(box.x, box.y, red)
    for i in sorted(base.invisible):
        if i in hits:
            outline(hits[i].x, hits[i].y, green)
        else:
            outline(*sheet.truth[i], purple, dashed=True)
    for x, y, label, hue in [
        (22, 550, "Baseline hit", blue), (317, 550, "Wrong baseline box", red),
        (22, 578, "Recovered with rotation bank", green),
        (317, 578, "Invisible miss (truth only)", purple),
    ]:
        cv2.line(canvas, (x, y - 5), (x + 17, y - 5), hue, 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (x + 25, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .49, ink, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f"Could not write {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="write the annotated example PNG")
    args = parser.parse_args()

    tune = prepare(TUNE_SEEDS)
    threshold = calibrate_baseline(tune)
    tune_base = [baseline(s, threshold) for s in tune]
    review = {source: calibrate_review(tune, tune_base, source, threshold)
              for source in ("upright", "rotated")}
    print(f"Tuning ({len(tune)} sheets): baseline {threshold:.2f}, "
          f"lower {review['upright']:.2f}, rotation bank {review['rotated']:.2f}")
    print_counts(collect(tune, tune_base, review).sum(axis=0))

    test = prepare(TEST_SEEDS)
    test_base = [baseline(s, threshold) for s in test]
    print(f"Test ({len(test)} sheets)")
    print_report(collect(test, test_base, review))
    if args.image:
        index = TEST_SEEDS.index(IMAGE_SEED)
        render_example(args.image, test[index], test_base[index], review["rotated"])


if __name__ == "__main__":
    main()
