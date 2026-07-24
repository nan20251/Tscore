"""Smoke tests for the tscore package."""

def test_import_tscore():
    import tscore
    assert hasattr(tscore, 'DeepDock_PPI')
    assert hasattr(tscore, 'ppi_train_loss')
    assert hasattr(tscore, 'ppi_score')
    assert isinstance(tscore.__version__, str)


def test_import_models_submodule():
    from tscore.models import DeepDock_PPI, ppi_train_loss, ppi_score
    assert DeepDock_PPI is not None
    assert callable(ppi_train_loss)
    assert callable(ppi_score)
