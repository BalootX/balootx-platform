import importlib.util


def test_quantiles_do_not_drop_incomplete_decisions(root):
    spec = importlib.util.spec_from_file_location('outcome_dd_benchmark', root / 'scripts/benchmark_outcome_dd.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [dict(mode='sun', trick=1, status='exact', wallSeconds=i/1000) for i in range(1, 20)]
    rows.append(dict(mode='sun', trick=1, status='incomplete', stopReason='maximum_seconds', wallSeconds=1))
    group, = module.summarize_rows(rows)
    assert group['p50Seconds'] == .010 and group['p95Seconds'] == .019
    assert group['counts'] == {'exact': 19, 'incomplete': 1}
    rows[-2].update(status='incomplete', stopReason='maximum_seconds', wallSeconds=1)
    group, = module.summarize_rows(rows)
    assert group['p50Seconds'] == .010 and group['p95Seconds'] is None
    rows[-1]['wallSeconds'] = .001
    group, = module.summarize_rows(rows)
    assert group['p50Seconds'] is None and group['p95Seconds'] is None
