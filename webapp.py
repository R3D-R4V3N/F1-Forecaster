from flask import (
    Flask,
    request,
    redirect,
    render_template_string,
    jsonify,
    url_for,
)
from threading import Thread
import yaml
import pandas as pd

from predictor.utils import parse_time_value, normalize_driver_name, format_time
from predictor.openf1_utils import (
    list_drivers,
    race_drivers,
    season_schedule,
    weather_summary,
    session_fastest_times,
    overtake_potential,
    slugify,
)
from predictor.model import load_training_data

app = Flask(__name__)

YEAR = 2025

try:
    DRIVER_OPTIONS = list_drivers(YEAR).to_dict("records")
except Exception as exc:  # pragma: no cover - network/dependency failures
    print(f"Failed to fetch drivers: {exc}")
    DRIVER_OPTIONS = []

try:
    SCHEDULE_OPTIONS = season_schedule(YEAR)
except Exception as exc:  # pragma: no cover - network/dependency failures
    print(f"Failed to fetch schedule: {exc}")
    SCHEDULE_OPTIONS = []


def prefetch_training_data(grand_prix: str) -> None:
    """Download historical data for a Grand Prix in the background."""

    def _task():
        years = list(range(2018, YEAR))
        try:
            load_training_data(years, grand_prix)
        except Exception as exc:  # pragma: no cover - network/dependency failures
            print(f"Prefetch failed for {grand_prix}: {exc}")

    print(f"Prefetching training data for {grand_prix}. This may take a few minutes...")
    Thread(target=_task, daemon=True).start()

FORM_TEMPLATE = """
<h2>🏁 Add Race Data</h2>
{% if message %}
  <p style="color: orange;">{{ message }}</p>
{% endif %}
<form method="post">
  <label>Grand Prix 🏟️:
    <select name="grand_prix" id="grand_prix" onchange="onGPChange()">
      {% for r in schedule_options %}
        <option value="{{ r.GrandPrix }}">{{ r.GrandPrix }}</option>
      {% endfor %}
    </select>
  </label><br>
  <label>Race key 🔑: <input name="race_name" id="race_name"></label><br>
  <fieldset>
    <legend>Weather 🌦️</legend>
    Air temp: <input name="air_temp" id="air_temp" value="0"><br>
    Track temp: <input name="track_temp" id="track_temp" value="0"><br>
    Rainfall (0-1): <input name="rainfall" id="rainfall" value="0"><br>
    Overtake potential: <input name="overtake_potential" value="0"><br>
  </fieldset>
  <table id="drivers">
    <tr><th>Driver 🚗</th><th>Q1</th><th>Q2</th><th>Q3</th><th>Quali</th></tr>
    {% for _ in range(6) %}
    <tr>
      <td>
        <select name="driver" onchange="onDriverChange(this)">
          {% for d in driver_options %}
            <option value="{{ d.Driver }}">{{ d.FullName }}</option>
          {% endfor %}
        </select>
      </td>
      <td><input name="q1"></td>
      <td><input name="q2"></td>
      <td><input name="q3"></td>
      <td><input name="quali"></td>
    </tr>
    {% endfor %}
  </table>
  <button type="button" onclick="addRow()">➕ Add Driver</button><br><br>
  <input type="submit" value="Save ✅">
</form>

<script>
function addRow() {
  const table = document.getElementById('drivers');
  const row = table.insertRow();
  row.innerHTML = table.rows[1].innerHTML;
}

function slugify(text) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
}

function onGPChange() {
  const gp = document.getElementById('grand_prix').value;
  document.getElementById('race_name').value = slugify(gp);
  if (!gp) return;
  fetch('/weather?gp=' + encodeURIComponent(gp))
    .then(r => r.json())
    .then(d => {
      document.getElementById('air_temp').value = d.air_temp || 0;
      document.getElementById('track_temp').value = d.track_temp || 0;
      document.getElementById('rainfall').value = d.rainfall || 0;
      document.querySelector('input[name="overtake_potential"]').value = d.overtake_potential || 0;
    });
  fetch('/drivers?gp=' + encodeURIComponent(gp))
    .then(r => r.json())
    .then(drivers => {
      const selects = document.querySelectorAll('#drivers select[name="driver"]');
      const options = drivers.map(d => `<option value="${d.Driver}">${d.FullName}</option>`).join('');
      selects.forEach(sel => {
        const current = sel.value;
        sel.innerHTML = options;
        if (Array.from(sel.options).some(o => o.value === current)) {
          sel.value = current;
        }
      });
    });
}

function onDriverChange(sel) {
  const gp = document.getElementById('grand_prix').value;
  if (!gp) return;
  const drv = sel.value;
  if (!drv) return;
  const row = sel.parentElement.parentElement;
  fetch('/times?gp=' + encodeURIComponent(gp) + '&driver=' + encodeURIComponent(drv))
    .then(r => r.json())
    .then(d => {
      row.querySelector('input[name="q1"]').value = d.q1 || '';
      row.querySelector('input[name="q2"]').value = d.q2 || '';
      row.querySelector('input[name="q3"]').value = d.q3 || '';
      row.querySelector('input[name="quali"]').value = d.quali || '';
    });
}
</script>
"""

@app.route('/', methods=['GET', 'POST'])
def add_race():
    if request.method == 'POST':
        gp = request.form['grand_prix']
        race_name = request.form.get('race_name') or slugify(gp)
        weather = {
            'air_temp': float(request.form.get('air_temp', 0)),
            'track_temp': float(request.form.get('track_temp', 0)),
            'rainfall': float(request.form.get('rainfall', 0)),
            'overtake_potential': float(request.form.get('overtake_potential', 0)),
        }
        if not request.form.get('air_temp'):
            weather.update(weather_summary(YEAR, gp))
            weather['rainfall'] = 1 if weather.get('rainfall', 0) > 0 else 0
            weather['overtake_potential'] = overtake_potential(YEAR - 1, gp)
        drivers = []
        codes = request.form.getlist('driver')
        q1s = request.form.getlist('q1')
        q2s = request.form.getlist('q2')
        q3s = request.form.getlist('q3')
        qualis = request.form.getlist('quali')
        for code, q1, q2, q3, quali in zip(codes, q1s, q2s, q3s, qualis):
            if not code:
                continue
            drivers.append({
                'Driver': normalize_driver_name(code),
                'Q1Time_s': parse_time_value(q1),
                'Q2Time_s': parse_time_value(q2),
                'Q3Time_s': parse_time_value(q3),
                'QualifyingTime_s': parse_time_value(quali),
            })
        try:
            with open('races.yaml', 'r') as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            data = {}
        data[race_name] = {'grand_prix': gp, 'drivers': drivers, 'weather': weather}
        with open('races.yaml', 'w') as f:
            yaml.safe_dump(data, f)
        # Prefetch historical data for training in the background
        prefetch_training_data(gp)
        return redirect(url_for('add_race', message='prefetch'))
    gp_selected = request.args.get('grand_prix', '')
    message_key = request.args.get('message', '')
    message_text = ''
    if message_key == 'prefetch':
        message_text = (
            'Downloading historical data in the background. '
            'This may take a few minutes.'
        )
    global SCHEDULE_OPTIONS
    if not SCHEDULE_OPTIONS:
        try:
            SCHEDULE_OPTIONS = season_schedule(YEAR)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to refresh schedule: {exc}")
            SCHEDULE_OPTIONS = []
    race_key = slugify(gp_selected) if gp_selected else ''
    weather = weather_summary(YEAR, gp_selected) if gp_selected else {
        'air_temp': 0,
        'track_temp': 0,
        'rainfall': 0,
    }
    return render_template_string(
        FORM_TEMPLATE,
        driver_options=DRIVER_OPTIONS,
        schedule_options=SCHEDULE_OPTIONS,
        gp=gp_selected,
        race_key=race_key,
        weather=weather,
        message=message_text,
    )


@app.route('/weather')
def get_weather():
    gp = request.args.get('gp', '')
    if not gp:
        return {}
    # Cache session times when a Grand Prix is selected so driver auto-fill is fast
    try:
        session_fastest_times(YEAR, gp)
    except Exception:
        pass
    weather = weather_summary(YEAR, gp)
    rain_flag = 1 if weather.get('rainfall', 0) > 0 else 0
    ot_potential = overtake_potential(YEAR - 1, gp)
    weather['rainfall'] = rain_flag
    weather['overtake_potential'] = ot_potential
    return weather


@app.route('/drivers')
def get_drivers():
    gp = request.args.get('gp', '')
    if not gp:
        return jsonify([])
    df = race_drivers(YEAR, gp)
    return jsonify(df.to_dict('records'))


@app.route('/times')
def get_times():
    gp = request.args.get('gp', '')
    driver = request.args.get('driver', '')
    if not gp or not driver:
        return jsonify({})
    df = session_fastest_times(YEAR, gp)
    row = df[df['Driver'] == driver]
    if row.empty:
        return jsonify({})
    row = row.iloc[0]
    return jsonify({
        'q1': format_time(row.get('Q1Time_s')),
        'q2': format_time(row.get('Q2Time_s')),
        'q3': format_time(row.get('Q3Time_s')),
        'quali': format_time(row.get('QualifyingTime_s')),
    })


def main():
    """Run a local web server for entering race data.

    Usage:
        python3 webapp.py
    """
    app.run(debug=True)


if __name__ == '__main__':
    main()
