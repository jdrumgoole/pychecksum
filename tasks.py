from invoke import task

@task
def test(c):
    c.run('pytest tests')

@task
def test_cli(c):
    c.run('python src/clichecksum.py --help', hide=True)

@task(test)
def build(c):
    c.run('poetry build')

@task(build)
def publish(c):
    c.run('poetry publish')


