import unittest

from rag_im.extract import html_to_text
from rag_im.sections import normalise, section_hash, split_text
from tests.support import fixture_text


class TestSectionSplitting(unittest.TestCase):
    def setUp(self):
        self.text = html_to_text(fixture_text("au_189_v1.html"), selector="main#content")
        self.sections = split_text(self.text)

    def test_section_count(self):
        self.assertEqual(len(self.sections), 7)

    def test_headings_and_levels(self):
        titles = [s.heading for s in self.sections]
        self.assertEqual(titles[0], "Skilled Independent visa (subclass 189)")
        self.assertEqual(self.sections[0].level, 1)
        self.assertIn("Points requirement", titles)
        points = next(s for s in self.sections if s.heading == "Points requirement")
        self.assertEqual(points.level, 2)

    def test_anchors_unique_and_slugged(self):
        anchors = [s.anchor for s in self.sections]
        self.assertEqual(len(anchors), len(set(anchors)))
        self.assertIn("points-requirement", anchors)
        self.assertIn("skilled-independent-visa-subclass-189", anchors)

    def test_body_excludes_heading(self):
        points = next(s for s in self.sections if s.heading == "Points requirement")
        self.assertNotIn("Points requirement", points.body)
        self.assertIn("65 points", points.body)

    def test_whitespace_reflow_keeps_hashes(self):
        # Simulate a CMS reflow: extra indentation, doubled spaces, blank lines
        # between blocks. Heading markers themselves stay intact.
        lines = []
        for line in self.text.splitlines():
            if not line.strip():
                continue
            if line.startswith("#"):
                lines.append(line)
            else:
                lines.append("    " + line.replace(" ", "   "))
        reflowed = "\n\n \n".join(lines)
        other = {s.anchor: s.hash for s in split_text(reflowed)}
        self.assertEqual(len(other), len(self.sections))
        for sec in self.sections:
            self.assertEqual(sec.hash, other[sec.anchor], sec.heading)

    def test_hash_stable_under_nonbreaking_spaces(self):
        reflowed = self.text.replace(" ", "\xa0")
        other = {s.anchor: s.hash for s in split_text(reflowed)}
        for sec in self.sections:
            self.assertEqual(sec.hash, other[sec.anchor], sec.heading)

    def test_hash_independent_of_heading_level(self):
        self.assertEqual(section_hash("Points", "65 points"), section_hash("Points", "  65\npoints "))

    def test_content_change_changes_hash(self):
        points = next(s for s in self.sections if s.heading == "Points requirement")
        self.assertNotEqual(points.hash, section_hash("Points requirement", "67 points"))

    def test_preamble_becomes_section(self):
        sections = split_text("Some introductory words.\n\n# Heading\n\nBody text.")
        self.assertEqual(sections[0].heading, "(preamble)")
        self.assertEqual(sections[0].level, 0)
        self.assertIn("introductory", sections[0].body)
        self.assertEqual(len(sections), 2)

    def test_empty_preamble_dropped(self):
        sections = split_text("\n\n# Heading\n\nBody")
        self.assertEqual(len(sections), 1)
        self.assertEqual(sections[0].heading, "Heading")

    def test_duplicate_headings_get_distinct_anchors(self):
        sections = split_text("# Notes\n\none\n\n# Notes\n\ntwo")
        self.assertEqual([s.anchor for s in sections], ["notes", "notes~2"])
        self.assertNotEqual(sections[0].hash, sections[1].hash)

    def test_normalise_collapses_whitespace(self):
        self.assertEqual(normalise("  a\n\tb   c "), "a b c")

    def test_empty_text(self):
        self.assertEqual(split_text(""), [])


if __name__ == "__main__":
    unittest.main()
