import os
import subprocess
import json
from pathlib import Path

from datetime import datetime, timedelta


cache_dir = Path('.cache').absolute()
aw_research_dir = Path("aw-research").absolute()


env = os.environ.copy()
env["PIPENV_IGNORE_VIRTUALENVS"] = "1"


def _read_qslang_cats():
    with open('./data/substance-categories.txt', 'r') as f:
        return [l.strip() for l in f.readlines()]


qslang_cats = _read_qslang_cats()


def _read_people_colocate():
    with open('./data/people-colocate.txt', 'r') as f:
        return [l.strip() for l in f.readlines()]


people_colocate = _read_people_colocate()

timeplot_cats = ['', 'Work', 'School', 'Programming', 'Finance', 'Media', 'Social Media', 'Communication', 'Uncategorized']

category_sunbursts_plots = [
    (timedelta(0), 'today'),
    (timedelta(days=7), 'last7d'),
    (timedelta(days=30), 'last30d'),
    (timedelta(days=90), 'last90d'),
]


def _build_category_sunburst():
    print("Building activity category sunbursts...")
    now = datetime.now()
    for offset, name in category_sunbursts_plots:
        dt = now - offset
        dt_str = dt.strftime("%Y-%m-%d")
        p = subprocess.run(f"pipenv run python3 -m aw_research classify --start {dt_str} summary_plot --save {cache_dir}/{name}-sunburst.png",
                           shell=True, cwd=aw_research_dir, env=env, capture_output=True)
        if p.stderr:
            print(str(p.stderr, "utf-8"))
        lines = str(p.stdout, "utf-8").split("\n")
        duration = next(l.strip().lstrip("Duration:").strip() for l in lines if "Duration" in l)
        with open(f'.cache/{name}-sunburst.json', 'w') as f:
            json.dump({"duration": duration}, f)
        print(f' - Built {name}')


def _build_category_timeplot():
    print("Building category timeplots...")
    now = datetime.now()
    dt = now - timedelta(days=90)
    dt_str = dt.strftime("%Y-%m-%d")
    for cat in timeplot_cats:
        p = subprocess.run(f"pipenv run python3 -m aw_research classify --start {dt_str} cat_plot '{cat}' --save '{cache_dir}/last60d-{cat}.png'",
                           shell=True, cwd=aw_research_dir, env=env, capture_output=True)
        if p.stderr:
            print(p.stderr)
        print(f' - Built {cat}')


def _build_location_plot():
    print("Building location plots...")
    now = datetime.now()
    dt = now - timedelta(days=90)
    dt_str = dt.strftime("%Y-%m-%d")
    for name in people_colocate:
        p = subprocess.run(f"pipenv run python3 scripts/location.py {name} --start {dt_str} --save {cache_dir}/last60d-location-{name}.png",
                           shell=True, env=env, capture_output=True)
        if p.stderr:
            print(p.stderr)
        print(f' - Built {name}')


def _read_metadata():
    import glob
    metadata = {}
    for filepath in glob.glob('.cache/*.json'):
        name = filepath.split("/")[-1].rstrip(".json")
        with open(filepath, "r") as f:
            metadata[name] = json.load(f)
    return metadata


def _build_qslang_plots():
    print("Building qslang plots...")
    dt = datetime.now() - timedelta(days=60)
    dt_str = dt.strftime("%Y-%m-%d")
    for cat in qslang_cats:
        p = subprocess.run(f"pipenv run python3 main.py --start {dt_str} --save ../.cache/last60d-substances-{cat.strip('#')}.png plot --count --daily --days '{cat}'",
                           shell=True, cwd="QSlang/", env=env, capture_output=True)
        if p.stderr:
            print(p.stderr)
        print(f' - Built {cat}')


def _build_nothing():
    pass


def build(what=None):
    _scope = globals()
    # print(_scope)

    if what is None:
        _build_category_sunburst()
        _build_category_timeplot()
        _build_location_plot()
        _build_qslang_plots()
    elif f"_build_{what}" in _scope:
        _scope[f"_build_{what}"]()
    else:
        raise Exception(f'unknown thing to generate: {what}')

    metadata = _read_metadata()

    css = """
        h1 {
            margin: 0;
            padding: 0;
        }

        img {
            mix-blend-mode: multiply;
        }

        summary {
            padding: 1em;
            background: #DDD;
        }

        details {
            background: #F5F5F5;
            margin-bottom: 0.5em;
            margin-right: 0.5em;
        }
    """

    return f"""
    <html>
        <style>{css}</style>
        <body>
            <div style="display: flex; flex-wrap: wrap;">
                {"".join(f'''
                    <div style="display: block">
                        <h1 style="text-transform: capitalize">{name}</h1>
                        <div>Duration: {metadata[name + "-sunburst"]["duration"]}</div>
                        <img src=".cache/{name}-sunburst.png" style="margin: -2em"/>
                    </div>
                    ''' for offset, name in category_sunbursts_plots)}
            </div>

            <div style="display: block">
                <h1>Last 60 days</h1>
                <div style="display: flex; flex-wrap: wrap;">

                    <details>
                        <summary>Activities</summary>
                        <div style="display: flex; flex-wrap: wrap;">
                            {'<br>'.join(f'<img src=".cache/last60d-{name}.png"/>' for name in timeplot_cats)}
                        </div>
                    </details>

                    <details>
                        <summary>Locations</summary>
                        <div style="display: flex; flex-wrap: wrap;">
                            {'<br>'.join(f'<img src=".cache/last60d-location-{name}.png"/>' for name in people_colocate)}
                        </div>
                    </details>

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


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--what')
    return parser.parse_args()


def main():
    import shutil
    args = _parse_args()
    if args.clear:
        shutil.rmtree(aw_research_dir / ".cache/joblib", ignore_errors=True)
    html = build(args.what)
    with open('dashboard.html', "w") as f:
        f.write(html)


if __name__ == "__main__":
    main()
