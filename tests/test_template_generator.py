"""Unit tests for TemplateGenerator."""

import os
import unittest
from grace.automation.template_generator import TemplateGenerator


class TestTemplateGenerator(unittest.TestCase):

    def test_render_text_template(self):
        png_path = TemplateGenerator.render_text_template("Play")
        self.assertIsNotNone(png_path)
        self.assertTrue(os.path.isfile(png_path))
        self.assertTrue(png_path.endswith(".png"))

    def test_render_empty_text_returns_none(self):
        self.assertIsNone(TemplateGenerator.render_text_template(""))
        self.assertIsNone(TemplateGenerator.render_text_template("   "))

    def test_render_text_templates_returns_multiple_variants(self):
        """Bug #5 fix: multiple font variants bridge the web-font mismatch."""
        paths = TemplateGenerator.render_text_templates("Play")
        self.assertGreaterEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(os.path.isfile(p))
            self.assertTrue(p.endswith(".png"))

    def test_render_text_templates_blank_returns_empty(self):
        self.assertEqual(TemplateGenerator.render_text_templates(""), [])
        self.assertEqual(TemplateGenerator.render_text_templates("   "), [])


if __name__ == "__main__":
    unittest.main()
