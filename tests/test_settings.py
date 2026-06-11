from config.settings import Settings


def test_default_embedding_model_is_multilingual(monkeypatch):
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    settings = Settings()

    assert settings.EMBEDDING_MODEL == (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
