import nox


@nox.session(reuse_venv=True)
def example(session):
    """Run the example scenario and plot the results."""
    session.install('-e', '.')
    session.run('python3', 'run-example.py')
