package main

import (
	"strings"
	"testing"
)

func TestExtractReferences(t *testing.T) {
	src := `from transformers import AutoModel, AutoTokenizer, pipeline
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

tok = AutoTokenizer.from_pretrained("bert-base-uncased/variant")
model = AutoModel.from_pretrained('openai/clip-vit-base-patch32')
clf = pipeline("sentiment-analysis", "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
clf2 = pipeline("text-generation", model="meta-llama/Llama-2-7b")
ds = load_dataset("stanfordnlp/imdb")
p = hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="config.json")
snap = snapshot_download(repo_id="black-forest-labs/FLUX.1-dev")
dl = hf_hub_download(repo_id="allenai/c4", filename="README.md", repo_type="dataset")
dl2 = hf_hub_download(repo_type="dataset", repo_id="allenai/soda", filename="README.md")
snapds = snapshot_download(repo_id="huggingfaceh4/no_robots", repo_type="dataset")
local = AutoModel.from_pretrained("./local-dir")
nocheck = AutoModel.from_pretrained("bert-base-uncased")
wrapped = AutoModel.from_pretrained(
    "facebook/bart-large",
    trust_remote_code=True,
)
clf3 = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b",
)
pos = hf_hub_download("openai-community/gpt2", "config.json")
possnap = snapshot_download("distilbert/distilgpt2")
posmulti = hf_hub_download(
    "bert-base-multilingual-cased/variant",
    "config.json",
)
posdataset = hf_hub_download("allenai/c4-positional", "README.md", repo_type="dataset")
`
	refs := extractReferences(src, "sample.py")

	want := map[string]repoKind{
		"bert-base-uncased/variant":                                  kindModel,
		"openai/clip-vit-base-patch32":                               kindModel,
		"distilbert/distilbert-base-uncased-finetuned-sst-2-english": kindModel,
		"meta-llama/Llama-2-7b":                                      kindModel,
		"stanfordnlp/imdb":                                           kindDataset,
		"runwayml/stable-diffusion-v1-5":                             kindModel,
		"black-forest-labs/FLUX.1-dev":                               kindModel,
		"allenai/c4":                                                 kindDataset,
		"allenai/soda":                                               kindDataset,
		"huggingfaceh4/no_robots":                                    kindDataset,
		"facebook/bart-large":                                        kindModel,
		"tiiuae/falcon-7b":                                           kindModel,
		"openai-community/gpt2":                                      kindModel,
		"distilbert/distilgpt2":                                      kindModel,
		"bert-base-multilingual-cased/variant":                       kindModel,
		"allenai/c4-positional":                                      kindDataset,
	}

	got := map[string]repoKind{}
	for _, r := range refs {
		got[r.id] = r.kind
		if r.source == "" {
			t.Errorf("ref %q has empty source", r.id)
		}
	}

	for id, kind := range want {
		gotKind, ok := got[id]
		if !ok {
			t.Errorf("expected to find ref %q, didn't", id)
			continue
		}
		if gotKind != kind {
			t.Errorf("ref %q: got kind %q, want %q", id, gotKind, kind)
		}
	}
	for id := range got {
		if _, ok := want[id]; !ok {
			t.Errorf("unexpected extra ref %q (e.g. local path or bare name wrongly matched)", id)
		}
	}
}

// TestExtractReferencesPositionalRepoID guards the fix for a false negative
// where hf_hub_download/snapshot_download calls using repo_id positionally
// (rather than as a repo_id= keyword) went completely undetected. repo_id is
// declared before the `*` that starts the keyword-only section in both
// functions' real signatures, so it's positional-or-keyword, and
// huggingface_hub's own docstrings call it that way — e.g.
// `hf_hub_download("openai-community/gpt2", "config.json", revision=revision)`
// in huggingface_hub/hf_api.py and `hf_hub_download('bert-base-cased',
// 'config.json', ...)` in huggingface_hub/errors.py. A hallucinated or
// typosquatted ID passed this way used to be silently skipped — "no
// references found" — rather than flagged.
func TestExtractReferencesPositionalRepoID(t *testing.T) {
	src := `cfg = hf_hub_download("0penai/this-definitely-does-not-exist-xyz", "config.json")
snap = snapshot_download("stanfordnlp/imdb-but-typo-xyz")
ds = hf_hub_download("allenai/c4-but-positional", "README.md", repo_type="dataset")

# a keyword-only first argument must NOT be mistaken for a positional
# repo_id just because it's followed by a quoted string later in the call
notrepo = hf_hub_download(filename="config.json", repo_id="allenai/c4")
`
	refs := extractReferences(src, "sample.py")

	want := map[string]repoKind{
		"0penai/this-definitely-does-not-exist-xyz": kindModel,
		"stanfordnlp/imdb-but-typo-xyz":             kindModel,
		"allenai/c4-but-positional":                 kindDataset,
		"allenai/c4":                                kindModel,
	}
	got := map[string]repoKind{}
	for _, r := range refs {
		got[r.id] = r.kind
	}
	for id, kind := range want {
		gotKind, ok := got[id]
		if !ok {
			t.Errorf("expected to find positional ref %q, didn't", id)
			continue
		}
		if gotKind != kind {
			t.Errorf("ref %q: got kind %q, want %q", id, gotKind, kind)
		}
	}
	for id := range got {
		if _, ok := want[id]; !ok {
			t.Errorf("unexpected extra ref %q", id)
		}
	}
	if _, ok := got["config.json"]; ok {
		t.Errorf("filename keyword value was wrongly extracted as a repo_id")
	}
}

// TestExtractReferencesIgnoresComments guards the fix for a false positive
// where a "#"-commented-out call was extracted and audited exactly like a
// live one. Commenting out a superseded from_pretrained/hf_hub_download call
// while iterating on its replacement is routine in real ML code — e.g.
// diffusers' own pipelines/stable_diffusion/convert_from_ckpt.py keeps:
//
//	# text_model = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
//	# text_model = CLIPTextModelWithProjection.from_pretrained(
//	#    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", projection_dim=1280
//	# )
//
// A hallucinated or typosquatted name left behind in a comment like that
// used to be reported as not_found alongside genuine live findings, even
// though the commented-out code never runs.
func TestExtractReferencesIgnoresComments(t *testing.T) {
	src := `# model = AutoModel.from_pretrained("openai/typo-model-that-does-not-exist-xyz123")
# text_model = CLIPTextModelWithProjection.from_pretrained(
#    "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", projection_dim=1280
# )

# a live call must still be found even though a comment precedes it
model = AutoModel.from_pretrained("meta-llama/still-live-xyz")

# a trailing same-line comment must not hide a live call that precedes it
model2 = AutoModel.from_pretrained("facebook/also-live-xyz")  # loads the base checkpoint

# a "#" inside a string literal is not a comment and must not truncate
# extraction of whatever (if anything) follows it on the same line
url = "https://example.com/model#section"
`
	refs := extractReferences(src, "sample.py")

	want := map[string]bool{
		"meta-llama/still-live-xyz": true,
		"facebook/also-live-xyz":    true,
	}
	got := map[string]bool{}
	for _, r := range refs {
		got[r.id] = true
	}
	for id := range want {
		if !got[id] {
			t.Errorf("expected to find live ref %q, didn't", id)
		}
	}
	for id := range got {
		if !want[id] {
			t.Errorf("unexpected ref %q: should have been ignored (commented-out or not a real call)", id)
		}
	}
}

// TestExtractReferencesIgnoresDocstringExamples guards the fix for a false
// positive where a from_pretrained/load_dataset call inside a docstring's
// illustrative "Example:" code block was extracted and audited exactly like
// a live call. Confirmed live against transformers' own source: dozens of
// its per-model modeling_*.py files carry a docstring Example built from a
// code-generation template that was never filled in with a real checkpoint,
// e.g. (modeling_llama4.py):
//
//	r"""
//	Example:
//
//	```python
//	>>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
//	```
//	"""
//
// "meta-llama4/Llama4-2-7b-hf" 401s on the real Hub API like any other
// nonexistent repo — it was never a real one — but that text is never
// executed as part of the program the docstring documents, exactly like a
// "#"-commented-out call. Without stripping it, scanning a library that
// uses this common template-placeholder convention in its own docstrings
// (transformers is not the only one) reports a flood of not_found findings
// that have nothing to do with the project's actual runtime behavior, and
// fails any build using hfaudit's default -fail-on.
func TestExtractReferencesIgnoresDocstringExamples(t *testing.T) {
	src := `class Llama4ForCausalLM:
    def forward(self):
        r"""
        Example:

        ` + "```python" + `
        >>> from transformers import AutoTokenizer, Llama4ForCausalLM

        >>> model = Llama4ForCausalLM.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama4/Llama4-2-7b-hf")
        ` + "```" + `
        """
        pass


# a live call must still be found even though a docstring example precedes it
model = AutoModel.from_pretrained("meta-llama/still-live-xyz")
`
	refs := extractReferences(src, "sample.py")

	want := map[string]bool{
		"meta-llama/still-live-xyz": true,
	}
	got := map[string]bool{}
	for _, r := range refs {
		got[r.id] = true
	}
	for id := range want {
		if !got[id] {
			t.Errorf("expected to find live ref %q, didn't", id)
		}
	}
	for id := range got {
		if !want[id] {
			t.Errorf("unexpected ref %q: should have been ignored (docstring example, not a real call)", id)
		}
	}
}

// TestExtractReferencesSecondPositionalFromPretrainedID guards the fix for a
// false negative where a from_pretrained call's second (and any further)
// leading positional argument was invisible to extraction entirely. Most
// from_pretrained-style classmethods take exactly one Hub ID, but the real
// setfit library's AbsaModel.from_pretrained takes two independent ones as
// separate positional parameters (model_id, then polarity_model_id) — see
// setfit/src/setfit/span/modeling.py:
//
//	def from_pretrained(cls, model_id: str, polarity_model_id: Optional[str] = None, ...):
//
// and real-world callers (e.g. huggingface/setfit's own tests/conftest.py)
// pass both positionally:
//
//	AbsaModel.from_pretrained(
//	    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
//	    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-polarity",
//	)
//
// Before the fix, hfaudit's from_pretrained pattern only ever captured the
// first quoted argument immediately after the opening paren — a hallucinated
// or typosquatted second positional ID (confirmed live: the real Hub API
// 401s on a made-up name exactly like any other nonexistent repo) produced
// zero findings and exit 0, completely invisible.
func TestExtractReferencesSecondPositionalFromPretrainedID(t *testing.T) {
	src := `from setfit import AbsaModel

model = AbsaModel.from_pretrained(
    "tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect",
    "tomaarsen/this-is-a-totally-fake-polarity-model-xyz123",
)

# a single-ID from_pretrained call must still work exactly as before
plain = AutoModel.from_pretrained("meta-llama/still-live-xyz")

# a second positional arg that ISN'T a bare quoted ID (e.g. a kwarg or a
# variable) must not be mistaken for a second Hub ID
single = AbsaModel.from_pretrained("tomaarsen/only-this-one-xyz", spacy_model="en_core_web_sm")
`
	refs := extractReferences(src, "sample.py")

	want := map[string]repoKind{
		"tomaarsen/setfit-absa-bge-small-en-v1.5-restaurants-aspect": kindModel,
		"tomaarsen/this-is-a-totally-fake-polarity-model-xyz123":     kindModel,
		"meta-llama/still-live-xyz":                                  kindModel,
		"tomaarsen/only-this-one-xyz":                                kindModel,
	}
	got := map[string]repoKind{}
	for _, r := range refs {
		got[r.id] = r.kind
	}
	for id, kind := range want {
		gotKind, ok := got[id]
		if !ok {
			t.Errorf("expected to find ref %q, didn't", id)
			continue
		}
		if gotKind != kind {
			t.Errorf("ref %q: got kind %q, want %q", id, gotKind, kind)
		}
	}
	for id := range got {
		if _, ok := want[id]; !ok {
			t.Errorf("unexpected extra ref %q (e.g. en_core_web_sm kwarg wrongly matched)", id)
		}
	}
}

func TestStripDeadPythonText(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"no comment", `x = 1`, `x = 1`},
		{"line comment", `x = 1  # set x`, `x = 1  `},
		{"whole-line comment", `# nothing here`, ``},
		{"hash in double-quoted string survives", `x = "a#b"  # real comment`, `x = "a#b"  `},
		{"hash in single-quoted string survives", `x = 'a#b'`, `x = 'a#b'`},
		{"escaped quote inside string doesn't end it early", `x = "a\"#b"  # c`, `x = "a\"#b"  `},
		// Triple-quoted content is blanked (not left verbatim, unlike an
		// ordinary quoted string): it's almost always a docstring, and a
		// docstring's illustrative "Example:" code is never executed as
		// part of the program it documents — see stripDeadPythonText's own
		// doc comment and TestExtractReferencesIgnoresDocstringExamples for
		// the real-world case this guards (transformers' per-model
		// docstrings showing a hallucination-shaped placeholder checkpoint
		// name like "meta-foo/Foo-2-7b-hf"). Newlines inside the blanked
		// span are kept so line numbers for whatever follows stay accurate.
		{
			"triple-quoted docstring body is blanked, newline preserved",
			"x = \"\"\"a # not a comment\nb\"\"\"  # real",
			"x = " + strings.Repeat(" ", 20) + "\n" + strings.Repeat(" ", 6),
		},
		{"backslash line continuation inside string keeps newline", "x = \"a\\\nb\"", "x = \"a\\\nb\""},
		{"multiple lines, newlines preserved", "a = 1  # one\nb = 2  # two\n", "a = 1  \nb = 2  \n"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := stripDeadPythonText(c.in)
			if got != c.want {
				t.Errorf("stripDeadPythonText(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

func TestIsSourceFile(t *testing.T) {
	cases := map[string]bool{
		"model.py":       true,
		"notebook.ipynb": true,
		"readme.md":      false,
		"script.js":      false,
		"noextension":    false,
		"dir/nested.py":  true,
	}
	for path, want := range cases {
		if got := isSourceFile(path); got != want {
			t.Errorf("isSourceFile(%q) = %v, want %v", path, got, want)
		}
	}
}
