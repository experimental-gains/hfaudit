package main

import "testing"

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
