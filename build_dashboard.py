import os
import subprocess
import json

from datetime import datetime, timedelta


def _build_category_sunbusts():
    print("Building activity category sunbursts...")
    env = os.environ.copy()
    env["PIPENV_IGNORE_VIRTUALENVS"] = "1"
    now = datetime.now()
    for dt, name in [
            (now, 'today'),
            (now - timedelta(days=7), 'last7d'),
            (now - timedelta(days=30), 'last30d'),
    ]:
        dt_str = dt.strftime("%Y-%m-%d")
        p = subprocess.run(f"pipenv run python3 -m aw_research classify --start {dt_str} summary_plot --save /home/erb/Programming/quantifiedme/.cache/{name}-sunburst.png",
                           shell=True, cwd="/home/erb/Programming/activitywatch/other/aw-research", env=env, capture_output=True)
        lines = str(p.stdout, "utf-8").split("\n")
        duration = next(l.strip().lstrip("Duration:").strip() for l in lines if "Duration" in l)
        with open(f'.cache/{name}-sunburst.json', 'w') as f:
            json.dump({"duration": duration}, f)


def _read_metadata():
    import glob
    metadata = {}
    for filepath in glob.glob('.cache/*.json'):
        name = filepath.split("/")[-1].rstrip(".json")
        with open(filepath, "r") as f:
            metadata[name] = json.load(f)
    return metadata


def _read_qslang_cats():
    with open('./data/substance-categories.txt', 'r') as f:
        return [l.strip() for l in f.readlines()]


qslang_cats = _read_qslang_cats()


def _build_qslang_plots():
    print("Building qslang plots...")
    env = os.environ.copy()
    env["PIPENV_IGNORE_VIRTUALENVS"] = "1"
    dt = datetime.now() - timedelta(days=60)
    dt_str = dt.strftime("%Y-%m-%d")
    for cat in qslang_cats:
        subprocess.run(f"pipenv run python3 main.py --start {dt_str} --save ../.cache/last60d-substances-{cat.strip('#')}.png plot --count --daily --days '{cat}'",
                       shell=True, cwd="QSlang/", env=env, capture_output=True)


def build():
    _build_category_sunbusts()
    _build_qslang_plots()
    metadata = _read_metadata()

    css = """
        h1 {
            margin: 0;
            padding: 0;
        }

        img {
            mix-blend-mode: multiply;
        }
    """

    return f"""
    <html>
        <style>{css}</style>
        <body>
            <div style="display: flex; flex-wrap: wrap;">
                <div style="display: block">
                    <h1>Today</h1>
                    <div>
                        Duration: {metadata['today-sunburst']['duration']}
                    </div>
                    <img src=".cache/today-sunburst.png" style="margin: -2.5em"/>
                </div>

                <div style="display: block">
                    <h1>Last 7 days</h1>
                    <div>
                        Duration: {metadata['last7d-sunburst']['duration']}
                    </div>
                    <img src=".cache/last7d-sunburst.png" style="margin: -2.5em"/>
                </div>

                <div style="display: block">
                    <h1>Last 30 days</h1>
                    <div>
                        Duration: {metadata['last30d-sunburst']['duration']}
                    </div>
                    <img src=".cache/last30d-sunburst.png" style="margin: -2.5em"/>
                </div>

                <div style="display: block">
                    <h1>Last 60 days</h1>
                    <details>
                        <summary>Drugs & Supplements</summary>
                        <div style="display: flex; flex-wrap: wrap;">
                            {'<br>'.join(f'<img src=".cache/last60d-substances-{cat.strip("#")}.png"/>' for cat in qslang_cats)}
                        </div>
                    </details>
                </div>
            </div>
        </body>
    </html>
    """


def main():
    html = build()
    with open('dashboard.html', "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
