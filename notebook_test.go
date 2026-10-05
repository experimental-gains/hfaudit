package main

import "testing"

// TestDecodeNotebookSourceUnescapesQuotes guards the fix for a bug where
// hfaudit scanned a .ipynb file's own raw JSON bytes directly instead of
// decoding it: nbformat stores a code cell's source as a JSON string, so
// every double-quote in real Python source inside it — the exact character
// idPattern's matching requires immediately surrounding a Hub ID — comes
// out as \" in that raw JSON text. A quote preceded by a backslash never
// satisfies a `["']` match at that position, so no double-quoted ID in any
// notebook could ever be extracted; only the (far rarer) single-quoted
// style happened to work, since JSON doesn't escape those. Confirmed live
// against setfit's own notebooks/text-classification.ipynb, which hit this
// on every one of its real from_pretrained calls.
func TestDecodeNotebookSourceUnescapesQuotes(t *testing.T) {
	raw := `{"cells": [` +
		`{"cell_type": "code", "source": ["model = SetFitModel.from_pretrained(\"lewtun/my-awesome-setfit-model\")\n"]},` +
		`{"cell_type": "code", "source": "ds = load_dataset(\"stanfordnlp/imdb\")"},` +
		`{"cell_type": "markdown", "source": ["See ` + "`" + `AutoModel.from_pretrained(\"should/not-be-scanned\")` + "`" + ` for details."]}` +
		`]}`

	text, ok := decodeNotebookSource([]byte(raw))
	if !ok {
		t.Fatalf("decodeNotebookSource returned ok=false on well-formed notebook JSON")
	}

	refs := extractReferences(text, "notebook.ipynb")
	got := map[string]repoKind{}
	for _, r := range refs {
		got[r.id] = r.kind
	}

	if got["lewtun/my-awesome-setfit-model"] != kindModel {
		t.Errorf("missing/wrong-kind model reference from code cell, got refs: %+v", refs)
	}
	if got["stanfordnlp/imdb"] != kindDataset {
		t.Errorf("missing/wrong-kind dataset reference from code cell, got refs: %+v", refs)
	}
	if _, found := got["should/not-be-scanned"]; found {
		t.Errorf("markdown-cell text was scanned as if it were live code: %+v", refs)
	}
}

// TestDecodeNotebookSourceMalformedJSON guards the fallback path: malformed
// or non-standard notebook JSON shouldn't make decodeNotebookSource panic or
// silently fabricate a result, it should just report ok=false so the caller
// falls back to scanning the raw bytes as plain text.
func TestDecodeNotebookSourceMalformedJSON(t *testing.T) {
	if _, ok := decodeNotebookSource([]byte("not even json")); ok {
		t.Error("decodeNotebookSource: ok = true on malformed JSON, want false")
	}
}

// TestDecodeCellSourceBothShapes guards decodeCellSource against the two
// source shapes nbformat actually allows: a single string, and a list of
// per-line strings joined back together with no extra separator (each line
// normally already carries its own trailing "\n").
func TestDecodeCellSourceBothShapes(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want string
	}{
		{"single string", `"a = 1\n"`, "a = 1\n"},
		{"list of lines", `["a = 1\n", "b = 2"]`, "a = 1\nb = 2"},
	}
	for _, c := range cases {
		got := decodeCellSource([]byte(c.raw))
		if got != c.want {
			t.Errorf("%s: decodeCellSource(%s) = %q, want %q", c.name, c.raw, got, c.want)
		}
	}
}
