from invoke import task

@task
def test(c):
    c.run('pytest tests')

@task
def test_cli(c):
    c.run('python src/clichecksum.py --help', hide=True)

@task(test, test_cli)
def build(c):
    c.run('poetry build')


