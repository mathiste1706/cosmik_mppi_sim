#!/usr/bin/env python3
"""Run the smoke check per human-arm frequency (optionally repeated) and tabulate how the robot copes.

Each run uses the URDF human with its arm moving as a sinusoid at the given
frequency. The per-target timeout is one arm period + 10 s (capped), so the
robot can wait for the arm to clear the way. Per run the table reports arrivals
at A/B, how the end effector crossed the arm's plane (above / below / beside the
forearm), the minimum robot-human clearance and the number of contact samples.

With --record-display, the X display (where RViz shows the simulation) is
recorded to video.mp4 in each run directory, with the run parameters overlaid.

Example:
    ./scripts/arm_frequency_sweep.py --output .runtime/sweep --frequencies 0.05 0.1 --repeats 3 \\
        --record-display :99
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess

DEFAULT_FREQUENCIES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def start_recording(display, path, caption):
    """ffmpeg capture of the X display with a caption burnt in; stop with stop_recording()."""
    text = caption.replace(':', r'\:').replace("'", '')
    return subprocess.Popen(
        ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'x11grab', '-video_size', '1920x1080',
         '-framerate', '10', '-i', display,
         '-vf', f"drawtext=fontfile={FONT}:text='{text}':x=30:y=30:fontsize=36:fontcolor=white:"
                "box=1:boxcolor=black@0.6:boxborderw=12",
         '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '30', '-pix_fmt', 'yuv420p', str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_recording(proc):
    try:
        proc.communicate(b'q', timeout=30)  # 'q' lets ffmpeg finalize the mp4.
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequencies', type=float, nargs='+', default=DEFAULT_FREQUENCIES, help='Hz')
    parser.add_argument('--repeats', type=int, default=1, help='runs per frequency')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--human-prediction', choices=['true', 'false'], default='true')
    parser.add_argument('--mppi-multimodal', choices=['true', 'false'], default='false')
    parser.add_argument('--min-seconds', type=float, default=50, help='minimum simulated seconds after the first goal')
    parser.add_argument('--max-timeout', type=float, default=120,
                        help='cap on the per-target timeout (one arm period + 10 s)')
    parser.add_argument('--min-timeout', type=float, default=0,
                        help='floor on the per-target timeout (the robot needs 15-20 s to leave its start pose)')
    parser.add_argument('--record-display', default=None, help='X display to record, e.g. :99 (RViz must run there)')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    smoke = Path(__file__).with_name('smoke.sh')
    config = f'multimodal={args.mppi_multimodal}  prediction={args.human_prediction}'

    rows = []
    for f in args.frequencies:
        # Let the robot wait a full arm cycle on a target, and give it at least two legs.
        timeout = min(max(1.0 / f + 10.0, args.min_timeout), args.max_timeout)
        seconds = max(args.min_seconds, 2.0 * timeout + 10.0)
        for k in range(1, args.repeats + 1):
            run_dir = args.output / (f'freq_{f:g}Hz' if args.repeats == 1 else f'freq_{f:g}Hz_run{k}')
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f'--- {f:g} Hz run {k}/{args.repeats}: target timeout {timeout:.0f} s, {seconds:.0f} s simulated '
                  f'-> {run_dir}', flush=True)
            recorder = None
            if args.record_display:
                recorder = start_recording(args.record_display, run_dir / 'video.mp4',
                                           f'bras humain: {f:g} Hz   run {k}/{args.repeats}   {config}')
            try:
                subprocess.run([str(smoke), '--human', 'true', '--human-model', 'urdf', '--human-motion', 'sine',
                                '--human-arm-frequency', str(f), '--human-prediction', args.human_prediction,
                                '--mppi-multimodal', args.mppi_multimodal, '--seconds', str(seconds),
                                '--target-timeout', str(timeout), '--wall-limit', str(1.5 * seconds + 120),
                                '--output', str(run_dir)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            finally:
                if recorder is not None:
                    stop_recording(recorder)
            result_path = run_dir / 'result.json'
            if not result_path.exists():
                rows.append({'frequency_hz': f, 'run': k, 'error': 'no result.json (see launch.log)'})
                continue
            r = json.loads(result_path.read_text())
            sides = [c['side'] for c in r.get('arm_crossings', [])]
            rows.append({
                'frequency_hz': f,
                'run': k,
                'target_timeout_s': timeout,
                'simulated_s': seconds,
                'smoke_pass': r['pass'],
                'arrivals_A': r['spatial_arrivals'].get('A', 0),
                'arrivals_B': r['spatial_arrivals'].get('B', 0),
                'crossed_above': sides.count('above'),
                'crossed_below': sides.count('below'),
                'crossed_beside': sides.count('beside'),
                'min_clearance_m': r.get('min_human_clearance_m'),
                'contact_samples': r.get('penetration_samples'),
                'samples': r.get('clearance_samples'),
                'timeouts': r['timeout_events'],
                'mppi_family_switches': len(r.get('mppi_family_switches', [])),
                'video': str(run_dir / 'video.mp4') if args.record_display else '',
            })
            print(json.dumps(rows[-1]), flush=True)
            # Rewrite the summary after every run so partial sweeps are usable.
            (args.output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')

    fields = []
    for row in rows:
        fields += [k for k in row if k not in fields]
    with (args.output / 'summary.csv').open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Summary: {args.output / "summary.csv"}')


if __name__ == '__main__':
    main()
