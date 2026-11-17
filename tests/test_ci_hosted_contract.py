"""Public build-policy contracts; private hygiene data is kept outside the tree."""
def test_hosted_ci_contract(root):
    import yaml
    workflow=yaml.safe_load((root/'.github/workflows/ci.yml').read_bytes())
    assert workflow['permissions']=={'contents':'read'}
    jobs=workflow['jobs']
    for job in jobs.values():
        assert job['runs-on']=='ubuntu-24.04'
        for step in job.get('steps',[]):
            if step.get('uses'):
                assert len(step['uses'].rsplit('@',1)[-1])==40
    assert jobs['ci-result']['name']=='CI result'
    assert jobs['ci-result']['needs']==['python-fast','native','sdks','media-release-smoke']
    assert jobs['ci-result']['if']=='always()'
    for name in ('media-release-smoke-run','python-backend-run','python-fast-run','python-heavy','native-run'):
        assert next(step for step in jobs[name]['steps'] if step.get('name')=='Install hosted native and media prerequisites')['timeout-minutes']==10
