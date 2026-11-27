import os
import unittest
from unittest.mock import patch

from rag_im import config
from rag_im.config import LLMSettings


class TestProviderProfiles(unittest.TestCase):
    def test_auto_without_key_is_bm25_only(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = config.llm_settings()
        self.assertEqual(settings.provider, "none")
        self.assertFalse(settings.enabled)
        self.assertFalse(settings.dense)
        self.assertEqual(settings.label, "no endpoint configured")

    def test_auto_with_openai_key_is_cloud(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            settings = config.llm_settings()
        self.assertEqual(settings.provider, "cloud")
        self.assertEqual(settings.base_url, "https://api.openai.com/v1")
        self.assertTrue(settings.enabled)
        self.assertTrue(settings.dense)

    def test_local_defaults_to_qwen_without_a_key(self):
        with patch.dict(os.environ, {"RAG_IM_PROVIDER": "local"}, clear=True):
            settings = config.llm_settings()
        self.assertEqual(settings.provider, "local")
        self.assertEqual(settings.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(settings.chat_model, "qwen3.5-2b")
        self.assertEqual(settings.embed_model, "qwen3.5-2b")
        self.assertTrue(settings.enabled)
        self.assertFalse(settings.dense)  # llama.cpp needs --embedding to serve vectors

    def test_local_overrides_and_embedding_opt_in(self):
        env = {
            "RAG_IM_PROVIDER": "local",
            "RAG_IM_LOCAL_BASE_URL": "http://127.0.0.1:9000/v1/",
            "RAG_IM_LOCAL_MODEL": "qwen-test",
            "RAG_IM_LOCAL_EMBED_MODEL": "embed-test",
            "RAG_IM_LOCAL_EMBEDDINGS": "1",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = config.llm_settings()
        self.assertEqual(settings.base_url, "http://127.0.0.1:9000/v1")
        self.assertEqual(settings.chat_model, "qwen-test")
        self.assertEqual(settings.embed_model, "embed-test")
        self.assertTrue(settings.dense)

    def test_hermes_proxy_defaults_to_a_free_model(self):
        with patch.dict(os.environ, {"RAG_IM_PROVIDER": "hermes"}, clear=True):
            settings = config.llm_settings()
        self.assertEqual(settings.base_url, "http://127.0.0.1:8645/v1")
        self.assertEqual(settings.chat_model, "upstage/solar-mini4:free")
        self.assertTrue(settings.enabled)
        self.assertFalse(settings.dense)

    def test_opencode_zen_profile_needs_no_key(self):
        with patch.dict(os.environ, {"RAG_IM_PROVIDER": "opencode"}, clear=True):
            settings = config.llm_settings()
        self.assertEqual(settings.base_url, "https://opencode.ai/zen/v1")
        self.assertEqual(settings.chat_model, "deepseek-v4.1-flash-free")
        self.assertTrue(settings.enabled)
        self.assertFalse(settings.dense)

    def test_none_disables_everything(self):
        with patch.dict(os.environ, {"RAG_IM_PROVIDER": "none"}, clear=True):
            settings = config.llm_settings()
        self.assertFalse(settings.enabled)
        self.assertFalse(settings.dense)
        self.assertEqual(settings.base_url, "")

    def test_explicit_provider_beats_environment(self):
        env = {"RAG_IM_PROVIDER": "cloud", "RAG_IM_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            settings = config.llm_settings("local")
        self.assertEqual(settings.provider, "local")

    def test_unknown_provider_is_a_clear_error(self):
        with patch.dict(os.environ, {"RAG_IM_PROVIDER": "wat"}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                config.llm_settings()
        self.assertIn("unknown provider", str(ctx.exception))

    def test_legacy_positional_settings_still_work(self):
        disabled = LLMSettings(None, "https://example.invalid/v1", "chat", "embed", 1)
        enabled = LLMSettings("sk-test", "https://example.invalid/v1", "chat", "embed", 1)
        self.assertFalse(disabled.enabled)
        self.assertFalse(disabled.dense)
        self.assertTrue(enabled.enabled)
        self.assertTrue(enabled.dense)


if __name__ == "__main__":
    unittest.main()
