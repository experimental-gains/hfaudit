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
spc = hf_hub_download(repo_id="adirik/OWL-ViT", repo_type="space", filename="assets/astronaut.png")
spc2 = hf_hub_download(repo_type="space", repo_id="ysharma/nougat", filename="input/nougat.pdf")
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
		"adirik/OWL-ViT":                                             kindSpace,
		"ysharma/nougat":                                             kindSpace,
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

// TestExtractReferencesResolvesSameFileConstant guards the gap closed by
// collectStringConstants/leadingPositionalIDs' identifier handling:
// from_pretrained(SOME_CONSTANT), where SOME_CONSTANT is assigned a Hub-ID
// literal earlier in the same file, used to be invisible to extraction
// entirely (the bare name failed leadingPositionalIDPattern's quoted-literal
// match, so the walk stopped with zero IDs found) — confirmed live in
// accelerate's own tests/fsdp/test_fsdp.py:
//
//	LLAMA_TESTING = "hf-internal-testing/tiny-random-LlamaForCausalLM"
//	...
//	model = AutoModel.from_pretrained(LLAMA_TESTING)
//
// A hallucinated or typosquatted ID assigned to a constant and referenced
// this way produced no finding and exit 0, same as every other gap this
// project's real-world-testing passes have closed.
func TestExtractReferencesResolvesSameFileConstant(t *testing.T) {
	src := `from transformers import AutoModel, AutoTokenizer

MODEL_ID = "meta-llama/this-definitely-does-not-exist-xyz-123"
TOKENIZER_ID: str = "openai/clip-vit-base-patch32"

model = AutoModel.from_pretrained(MODEL_ID)
tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)

# an undefined/unresolvable name must not produce a finding — it isn't a
# simple same-file literal assignment, so it stays invisible exactly like
# before this fix, not guessed at.
other = AutoModel.from_pretrained(SOME_IMPORTED_CONSTANT)

# a name that happens to be a keyword argument's own name (NAME=value)
# must never be mistaken for a bare positional identifier reference.
kw = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_ID)
`
	refs := extractReferences(src, "sample.py")

	want := map[string]repoKind{
		"meta-llama/this-definitely-does-not-exist-xyz-123": kindModel,
		"openai/clip-vit-base-patch32":                      kindModel,
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
			t.Errorf("unexpected extra ref %q (e.g. an unresolvable name wrongly matched)", id)
		}
	}
}

// TestExtractReferencesResolvesSameFileConstantOtherCallShapes guards a gap
// left behind by the fix TestExtractReferencesResolvesSameFileConstant
// covers: same-file constant resolution (collectStringConstants) was wired
// into leadingPositionalIDs for from_pretrained's positional arguments only.
// pipeline's model= keyword and second positional argument, load_dataset's
// first positional argument, and hf_hub_download/snapshot_download's
// repo_id= keyword and positional argument all independently required a
// quoted literal directly at the call site — a same-file constant
// referenced through any of those four other call shapes was completely
// invisible to extraction, exactly the same silent gap, just in four more
// places. Confirmed live in huggingface/datasets' own tests/test_load.py:
//
//	SAMPLE_DATASET_IDENTIFIER3 = "hf-internal-testing/multi_dir_dataset"
//	...
//	dataset = load_dataset(SAMPLE_DATASET_IDENTIFIER3)
//
// Pre-fix, running hfaudit against that real file found only one of its
// seven genuine Hub references (the one bare quoted literal); the other
// six, all passed through a same-file constant, produced no finding.
func TestExtractReferencesResolvesSameFileConstantOtherCallShapes(t *testing.T) {
	src := `from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download

MODEL_KW_ID = "meta-llama/fake-pipeline-model-keyword-xyz"
MODEL_POS_ID = "meta-llama/fake-pipeline-second-positional-xyz"
DATASET_ID = "allenai/fake-load-dataset-constant-xyz"
REPO_KW_ID = "meta-llama/fake-hf-hub-download-keyword-xyz"
REPO_POS_ID = "meta-llama/fake-hf-hub-download-positional-xyz"
SNAPSHOT_ID = "meta-llama/fake-snapshot-download-positional-xyz"
FILENAME_ID = "meta-llama/this-is-a-filename-value-not-a-repo-id-xyz"

clf = pipeline("text-classification", model=MODEL_KW_ID)
clf2 = pipeline("text-classification", MODEL_POS_ID)
ds = load_dataset(DATASET_ID)
p = hf_hub_download(repo_id=REPO_KW_ID, filename="config.json")
p2 = hf_hub_download(REPO_POS_ID, "config.json")
snap = snapshot_download(SNAPSHOT_ID)

# an unresolvable name must stay invisible, same as from_pretrained's own
# rule — not guessed at, exactly like a config lookup or CLI argument.
other = pipeline("text-classification", model=some_unresolvable_var)

# a name that's actually a keyword argument's own name (NAME=value) must
# never be mistaken for a bare positional identifier reference.
notrepo = hf_hub_download(repo_id="allenai/c4", filename=FILENAME_ID)
`
	refs := extractReferences(src, "sample.py")

	want := map[string]repoKind{
		"meta-llama/fake-pipeline-model-keyword-xyz":       kindModel,
		"meta-llama/fake-pipeline-second-positional-xyz":   kindModel,
		"allenai/fake-load-dataset-constant-xyz":           kindDataset,
		"meta-llama/fake-hf-hub-download-keyword-xyz":      kindModel,
		"meta-llama/fake-hf-hub-download-positional-xyz":   kindModel,
		"meta-llama/fake-snapshot-download-positional-xyz": kindModel,
		"allenai/c4": kindModel,
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
			t.Errorf("unexpected extra ref %q (e.g. an unresolvable name or a filename= value wrongly matched)", id)
		}
	}
}

// TestCollectStringConstants exercises collectStringConstants directly,
// including shapes it must NOT treat as a constant assignment: anything
// that isn't a bare literal alone on its line (an expression built from
// concatenation/an f-string, or a tuple-assignment line with trailing
// content after the literal).
func TestCollectStringConstants(t *testing.T) {
	src := `MODEL_ID = "org/name"
TYPED_ID: str = "org/typed-name"
NOT_A_CONST = "org/" + "concat"
TUPLE_LINE = "org/tuple", "org/other"
   INDENTED = "org/indented"
`
	got := collectStringConstants(stripDeadPythonText(src))
	want := map[string]string{
		"MODEL_ID": "org/name",
		"TYPED_ID": "org/typed-name",
		"INDENTED": "org/indented",
	}
	for name, id := range want {
		if got[name] != id {
			t.Errorf("collectStringConstants[%q] = %q, want %q", name, got[name], id)
		}
	}
	for _, bad := range []string{"NOT_A_CONST", "TUPLE_LINE"} {
		if _, ok := got[bad]; ok {
			t.Errorf("collectStringConstants wrongly captured %q (not a bare literal assignment)", bad)
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
