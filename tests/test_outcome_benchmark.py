import json
import csv

import yaml
import pytest

from scripts import benchmark_outcome_card_policies as benchmark


@pytest.mark.parametrize('reasons', [{}, {'campaign_time_budget': 1},
    {'outcome_miss': 2, 'unexpected_game_error:AssertionError': 1, 'unexpected_game_error:ValueError': 2}])
def test_volume_benchmark_exit_status_keeps_reason_counts(root, tmp_path, monkeypatch, capsys, reasons):
    monkeypatch.syspath_prepend(str(root/'scripts'))
    from scripts import benchmark_outcome_volume as volume
    monkeypatch.setattr('sys.argv', ['benchmark_outcome_volume.py', '--case', 'big-swing',
        '--policy', 'optimal', '--workers', '1', '--out', str(tmp_path/'volume')])
    monkeypatch.setattr(volume, 'generate_outcomes', lambda *args, **kwargs:
        dict(status='partial', rejectReasons=reasons))
    if any(reason.startswith('unexpected_game_error:') for reason in reasons):
        with pytest.raises(SystemExit) as exited:
            volume.main()
        assert exited.value.code == 1
    else:
        assert volume.main() is None
    summary = json.loads(capsys.readouterr().out)
    assert summary == dict(case='big-swing', policy='optimal', status='partial', rejectReasons=reasons)


def test_round_latency_summary_keeps_incomplete_rounds_censored_and_exports_csv(tmp_path, monkeypatch):
    output = tmp_path/'latency.json'
    from types import SimpleNamespace
    args = SimpleNamespace(rounds=3, seed=10, threads=4, out=output)
    monkeypatch.setattr(benchmark, 'arguments', lambda: args)
    def measure(args):
        return dict(mode='sun', policy='informed', seed=args.seed, status='complete' if args.seed < 12 else 'incomplete',
            fullRoundCompleted=args.seed < 12, effectiveThreads=4, wallSeconds=args.seed-9, cpuSeconds=args.seed-8)
    monkeypatch.setattr(benchmark, 'measure_round', measure)
    benchmark.main()
    result = json.loads(output.read_text(encoding='utf-8'))
    assert [r['seed'] for r in result['rounds']] == [10, 11, 12]
    assert result['completeRounds'] == 2 and result['incompleteRounds'] == 1
    assert result['allRoundP95Censored'] is True
    assert result['completeRoundWall'] == dict(p50=1, p95=2)
    assert len(list(csv.DictReader(output.with_suffix('.csv').open(encoding='utf-8')))) == 3


