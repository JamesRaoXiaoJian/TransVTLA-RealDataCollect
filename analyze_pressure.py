"""Analyze pressure data across all sessions based on channel mapping."""

from pathlib import Path
import csv
import statistics

# Channel Mapping
LEFT_CHANNEL = 51
RIGHT_CHANNEL = 50
LEFT_MATRIX_CHANNELS = [[63, 60, 57], [64, 61, 58], [49, 62, 59]]
RIGHT_MATRIX_CHANNELS = [[47, 44, 41], [48, 45, 42], [33, 46, 43]]


def load_pressure_csv(pressure_dir: Path) -> list[list[int]]:
    """Load pressure.csv -> list of 64 int values per row."""
    csv_path = pressure_dir / "pressure.csv"
    if not csv_path.exists():
        return []
    rows: list[list[int]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            try:
                vals = [int(x) for x in row[1:65]]
                if len(vals) == 64:
                    rows.append(vals)
            except (ValueError, IndexError):
                pass
    return rows


def get_channel_value(row: list[int], channel: int) -> int:
    """Get value from a specific channel (1-indexed)."""
    return row[channel - 1] if 0 <= channel - 1 < len(row) else 0


def analyze_sessions(base_path: Path) -> dict:
    """Analyze pressure data across all sessions."""
    # Find all session directories
    sessions = []
    for dji_dir in base_path.rglob("dji"):
        session_dir = dji_dir.parent
        if (session_dir / "pressure").is_dir():
            sessions.append(session_dir)

    sessions.sort(key=lambda p: str(p))

    # Statistics containers
    left_channel_values = []
    right_channel_values = []
    left_matrix_values = {ch: [] for row in LEFT_MATRIX_CHANNELS for ch in row}
    right_matrix_values = {ch: [] for row in RIGHT_MATRIX_CHANNELS for ch in row}

    session_stats = []

    for session in sessions:
        pressure_data = load_pressure_csv(session / "pressure")
        if not pressure_data:
            continue

        session_name = session.parent.name + "/" + session.name

        # Get values for each frame in this session
        session_left_ch = []
        session_right_ch = []
        session_left_mat = {ch: [] for ch in left_matrix_values}
        session_right_mat = {ch: [] for ch in right_matrix_values}

        for row in pressure_data:
            # Left channel
            left_val = get_channel_value(row, LEFT_CHANNEL)
            left_channel_values.append(left_val)
            session_left_ch.append(left_val)

            # Right channel
            right_val = get_channel_value(row, RIGHT_CHANNEL)
            right_channel_values.append(right_val)
            session_right_ch.append(right_val)

            # Left matrix channels
            for ch_list in LEFT_MATRIX_CHANNELS:
                for ch in ch_list:
                    val = get_channel_value(row, ch)
                    left_matrix_values[ch].append(val)
                    session_left_mat[ch].append(val)

            # Right matrix channels
            for ch_list in RIGHT_MATRIX_CHANNELS:
                for ch in ch_list:
                    val = get_channel_value(row, ch)
                    right_matrix_values[ch].append(val)
                    session_right_mat[ch].append(val)

        # Session-level statistics
        session_stat = {
            "name": session_name,
            "frames": len(pressure_data),
            "left_channel": {
                "mean": statistics.mean(session_left_ch) if session_left_ch else 0,
                "max": max(session_left_ch) if session_left_ch else 0,
                "min": min(session_left_ch) if session_left_ch else 0,
                "std": statistics.stdev(session_left_ch) if len(session_left_ch) > 1 else 0
            },
            "right_channel": {
                "mean": statistics.mean(session_right_ch) if session_right_ch else 0,
                "max": max(session_right_ch) if session_right_ch else 0,
                "min": min(session_right_ch) if session_right_ch else 0,
                "std": statistics.stdev(session_right_ch) if len(session_right_ch) > 1 else 0
            }
        }
        session_stats.append(session_stat)

    # Overall statistics
    results = {
        "total_sessions": len(session_stats),
        "total_frames": len(left_channel_values),
        "left_channel": {
            "channel": LEFT_CHANNEL,
            "mean": statistics.mean(left_channel_values) if left_channel_values else 0,
            "max": max(left_channel_values) if left_channel_values else 0,
            "min": min(left_channel_values) if left_channel_values else 0,
            "std": statistics.stdev(left_channel_values) if len(left_channel_values) > 1 else 0
        },
        "right_channel": {
            "channel": RIGHT_CHANNEL,
            "mean": statistics.mean(right_channel_values) if right_channel_values else 0,
            "max": max(right_channel_values) if right_channel_values else 0,
            "min": min(right_channel_values) if right_channel_values else 0,
            "std": statistics.stdev(right_channel_values) if len(right_channel_values) > 1 else 0
        },
        "left_matrix": {},
        "right_matrix": {},
        "sessions": session_stats
    }

    # Left matrix statistics
    for ch_list in LEFT_MATRIX_CHANNELS:
        for ch in ch_list:
            vals = left_matrix_values[ch]
            results["left_matrix"][ch] = {
                "mean": statistics.mean(vals) if vals else 0,
                "max": max(vals) if vals else 0,
                "min": min(vals) if vals else 0,
                "std": statistics.stdev(vals) if len(vals) > 1 else 0
            }

    # Right matrix statistics
    for ch_list in RIGHT_MATRIX_CHANNELS:
        for ch in ch_list:
            vals = right_matrix_values[ch]
            results["right_matrix"][ch] = {
                "mean": statistics.mean(vals) if vals else 0,
                "max": max(vals) if vals else 0,
                "min": min(vals) if vals else 0,
                "std": statistics.stdev(vals) if len(vals) > 1 else 0
            }

    return results


def format_matrix(matrix_channels: list[list[int]], stats: dict) -> str:
    """Format matrix statistics as a 3x3 grid."""
    lines = []
    for row in matrix_channels:
        row_vals = []
        for ch in row:
            s = stats[ch]
            row_vals.append(f"Ch{ch:2d}: mean={s['mean']:6.1f}, max={s['max']:5d}, min={s['min']:5d}")
        lines.append("  | ".join(row_vals))
    return "\n".join(lines)


def main():
    base_path = Path("sessions")
    if not base_path.exists():
        print("Error: sessions folder not found")
        return

    print("Analyzing pressure data across all sessions...")
    print("=" * 80)

    results = analyze_sessions(base_path)

    print(f"\nTotal Sessions: {results['total_sessions']}")
    print(f"Total Frames: {results['total_frames']}")

    # Left channel
    print("\n" + "=" * 80)
    print("LEFT CHANNEL (Channel {})".format(results['left_channel']['channel']))
    print("-" * 80)
    lc = results['left_channel']
    print(f"  Mean:  {lc['mean']:.2f}")
    print(f"  Max:   {lc['max']}")
    print(f"  Min:   {lc['min']}")
    print(f"  Std:   {lc['std']:.2f}")

    # Right channel
    print("\n" + "=" * 80)
    print("RIGHT CHANNEL (Channel {})".format(results['right_channel']['channel']))
    print("-" * 80)
    rc = results['right_channel']
    print(f"  Mean:  {rc['mean']:.2f}")
    print(f"  Max:   {rc['max']}")
    print(f"  Min:   {rc['min']}")
    print(f"  Std:   {rc['std']:.2f}")

    # Left matrix
    print("\n" + "=" * 80)
    print("LEFT 3x3 MATRIX")
    print("-" * 80)
    print("Layout:")
    print(format_matrix(LEFT_MATRIX_CHANNELS, results['left_matrix']))

    # Right matrix
    print("\n" + "=" * 80)
    print("RIGHT 3x3 MATRIX")
    print("-" * 80)
    print("Layout:")
    print(format_matrix(RIGHT_MATRIX_CHANNELS, results['right_matrix']))

    # Per-session summary
    print("\n" + "=" * 80)
    print("PER-SESSION SUMMARY")
    print("-" * 80)
    print(f"{'Session':<40} {'Frames':>8} {'Left Mean':>10} {'Left Max':>10} {'Right Mean':>11} {'Right Max':>10}")
    print("-" * 80)
    for s in results['sessions']:
        print(f"{s['name']:<40} {s['frames']:>8} {s['left_channel']['mean']:>10.1f} {s['left_channel']['max']:>10} {s['right_channel']['mean']:>11.1f} {s['right_channel']['max']:>10}")


if __name__ == "__main__":
    main()
