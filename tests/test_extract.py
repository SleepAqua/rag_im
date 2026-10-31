import unittest

from rag_im.extract import html_to_text
from tests.support import fixture_text


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.text = html_to_text(fixture_text("au_189_v1.html"), selector="main#content")

    def test_drops_scripts(self):
        self.assertNotIn("dataLayer", self.text)
        self.assertNotIn("analytics.js", self.text)
        self.assertNotIn("page_view", self.text)

    def test_drops_styles(self):
        self.assertNotIn("display: none", self.text)
        self.assertNotIn("gov-banner", self.text)

    def test_drops_navigation(self):
        self.assertNotIn("Visa finder", self.text)
        self.assertNotIn("Visa listing", self.text)

    def test_drops_header(self):
        self.assertNotIn("Search the website", self.text)
        self.assertNotIn("Department of Home Affairs", self.text)

    def test_drops_footer(self):
        self.assertNotIn("Acknowledgement of country", self.text)
        self.assertNotIn("Accessibility", self.text)

    def test_drops_aside(self):
        self.assertNotIn("Related visas", self.text)
        self.assertNotIn("subclass 190", self.text)

    def test_preserves_heading_structure(self):
        self.assertIn("# Skilled Independent visa (subclass 189)", self.text)
        self.assertIn("## Points requirement", self.text)
        self.assertIn("## Visa cost", self.text)

    def test_preserves_body_text(self):
        self.assertIn("AUD 4,765", self.text)
        self.assertIn("65 points", self.text)

    def test_lists_become_bullets(self):
        block = [line for line in self.text.splitlines() if "under 45" in line]
        self.assertTrue(block and block[0].startswith("- "), block)

    def test_selector_scopes_to_content(self):
        html = '<html><body><nav>nav noise</nav><div id="other">Ignore me</div><main id="c"><h1>Keep</h1><p>Body</p></main></body></html>'
        text = html_to_text(html, selector="#c")
        self.assertIn("Keep", text)
        self.assertNotIn("Ignore me", text)

    def test_missing_selector_falls_back_to_document(self):
        html = "<html><body><main><h1>Keep</h1><p>Body</p></main></body></html>"
        text = html_to_text(html, selector="#does-not-exist")
        self.assertIn("Keep", text)
        self.assertIn("Body", text)

    def test_invalid_selector_falls_back(self):
        html = "<html><body><p>Body</p></body></html>"
        self.assertIn("Body", html_to_text(html, selector="!!not a selector!!"))

    def test_empty_html(self):
        self.assertEqual(html_to_text(""), "")
        self.assertEqual(html_to_text("   "), "")

    def test_duplicate_nested_text_not_repeated(self):
        html = '<html><body><div class="card"><div class="card"><p>Same</p></div></div></body></html>'
        self.assertEqual(html_to_text(html).count("Same"), 1)


if __name__ == "__main__":
    unittest.main()
