package main

import (
	"fmt"
	"regexp"
	"strings"
)

// repoRef is a Hugging Face Hub ID found in source text, with the kind the
// call site implies and where it came from (for the human-readable report).
type repoRef struct {
	id     string
	kind   repoKind
	source string
}

// idPattern matches a Hugging Face repo ID: exactly one "/" separating an
// org/user segment from a name segment, both drawn from the character set
// the Hub actually allows (letters, digits, '-', '_', '.'). This deliberately
// excludes local filesystem paths with more than one slash and plain
// filenames with no slash at all, which from_pretrained also accepts (e.g.
// "./my-model" or "bert-base-uncased" without a namespace) — those aren't
// Hub IDs this tool can look up.
const idPattern = `[A-Za-z0-9][A-Za-z0-9_.\-]*/[A-Za-z0-9][A-Za-z0-9_.\-]*`

var extractPatterns = []struct {
	re   *regexp.Regexp
	kind repoKind
}{
	// Any `<something>.from_pretrained("org/name")` — covers AutoModel,
	// AutoTokenizer, every task-specific Auto* class, and third-party
	// wrappers (SentenceTransformer.from_pretrained, CLIPModel, ...).
	{regexp.MustCompile(`\.from_pretrained\(\s*["'](` + idPattern + `)["']`), kindModel},
	// pipeline("task", "org/name") or pipeline("task", model="org/name") —
	// the model ID can be positional (2nd arg) or the model= keyword.
	{regexp.MustCompile(`\bpipeline\(\s*["'][^"']*["']\s*,\s*["'](` + idPattern + `)["']`), kindModel},
	{regexp.MustCompile(`\bpipeline\([^)]*?\bmodel\s*=\s*["'](` + idPattern + `)["']`), kindModel},
	// load_dataset("org/name")
	{regexp.MustCompile(`\bload_dataset\(\s*["'](` + idPattern + `)["']`), kindDataset},
}

// hfHubDownloadCallPattern matches an hf_hub_download(...)/snapshot_download(...)
// call's whole argument list as one blob, rather than jumping straight to
// repo_id= the way extractPatterns' other entries do. Both functions also
// take a repo_type= keyword telling the Hub whether repo_id names a model or
// a dataset (or a Space, which this tool doesn't check), and repo_type can
// legally appear either before or after repo_id since both are keyword
// arguments — capturing the whole call first lets repoTypeArgPattern find it
// regardless of order.
var hfHubDownloadCallPattern = regexp.MustCompile(`\b(?:hf_hub_download|snapshot_download)\(([^)]*)\)`)

var repoIDArgPattern = regexp.MustCompile(`\brepo_id\s*=\s*["'](` + idPattern + `)["']`)

// positionalRepoIDArgPattern matches repo_id passed positionally rather than
// as a repo_id= keyword. repo_id is hf_hub_download's and snapshot_download's
// first parameter and is positional-or-keyword (declared before the `*` that
// starts the keyword-only section), and huggingface_hub's own docstrings
// call it that way: `hf_hub_download("openai-community/gpt2", "config.json",
// revision=revision)` (huggingface_hub/hf_api.py) and
// `hf_hub_download('bert-base-cased', 'config.json', ...)`
// (huggingface_hub/errors.py) — this is routine, documented usage, not an
// obscure corner. Python syntax requires positional arguments to precede any
// keyword arguments in a call, so a positional repo_id, when present, is
// always the first token in the argument list; anchoring to the start of
// the captured args blob (which begins right after the call's own open
// paren) finds it without also matching a quoted string that belongs to a
// later keyword argument.
var positionalRepoIDArgPattern = regexp.MustCompile(`^\s*["'](` + idPattern + `)["']`)

// repoTypeArgPattern matches the repo_type= keyword argument. The same ID
// string can be a real model under one kind and nonexistent under another,
// so getting this wrong means checking the wrong Hub endpoint entirely: a
// real, existing dataset (e.g. allenai/c4, confirmed 200 on
// /api/datasets/allenai/c4) looked up as a model (401 on /api/models/) comes
// back a false "not_found" hallucination.
var repoTypeArgPattern = regexp.MustCompile(`\brepo_type\s*=\s*["'](\w+)["']`)

// extractReferences scans source text for Hugging Face Hub references and
// reports each match's 1-based line number via "label:line" in source.
//
// Matching runs against the whole text rather than line-by-line: real-world
// calls are routinely wrapped across multiple lines (black/ruff formatting,
// or just extra kwargs like trust_remote_code=True), putting the opening
// "from_pretrained(" and the quoted ID on different lines. Per-line matching
// would silently miss every one of those — a false negative in a tool whose
// entire job is catching bad IDs.
//
// Python "#" comments are stripped first (for .py text; .ipynb files store
// source as JSON-escaped strings where a bare "#" doesn't delimit a real
// line, so they're left alone — see stripPythonComments). Commented-out
// calls are routine in real ML code — e.g. diffusers' own
// pipelines/stable_diffusion/convert_from_ckpt.py keeps a superseded
// from_pretrained call around as a comment while iterating on the
// replacement — and the whole point of this tool is flagging IDs the real
// program will actually try to fetch, not dead code. Without this, a typo
// or hallucinated name left behind in a commented-out call is reported
// exactly like a live one, a false positive this tool has no business
// raising.
func extractReferences(text, label string) []repoRef {
	if !strings.HasSuffix(label, ".ipynb") {
		text = stripPythonComments(text)
	}
	var refs []repoRef
	for _, p := range extractPatterns {
		for _, m := range p.re.FindAllStringSubmatchIndex(text, -1) {
			id := text[m[2]:m[3]]
			line := 1 + strings.Count(text[:m[0]], "\n")
			refs = append(refs, repoRef{
				id:     id,
				kind:   p.kind,
				source: fmt.Sprintf("%s:%d", label, line),
			})
		}
	}
	for _, m := range hfHubDownloadCallPattern.FindAllStringSubmatchIndex(text, -1) {
		args := text[m[2]:m[3]]
		idm := repoIDArgPattern.FindStringSubmatch(args)
		if idm == nil {
			idm = positionalRepoIDArgPattern.FindStringSubmatch(args)
		}
		if idm == nil {
			continue
		}
		kind := kindModel
		if tm := repoTypeArgPattern.FindStringSubmatch(args); tm != nil && tm[1] == "dataset" {
			kind = kindDataset
		}
		line := 1 + strings.Count(text[:m[0]], "\n")
		refs = append(refs, repoRef{
			id:     idm[1],
			kind:   kind,
			source: fmt.Sprintf("%s:%d", label, line),
		})
	}
	return refs
}

// stripPythonComments removes Python "#"-to-end-of-line comments from text,
// while leaving string literal contents (including anything that happens to
// contain a "#", like an f-string or a URL) untouched, and without disturbing
// line numbers: comment bytes are dropped but every newline byte is kept, so
// 1+strings.Count(text[:i], "\n") against the result still lines up with the
// original source.
//
// This is a small hand-rolled Python lexer, not a full one — it only needs
// to track enough state (single/double/triple-quoted strings, backslash
// escapes) to tell a real comment-starting "#" apart from one sitting inside
// a string. String prefixes (r"...", f"...", b"...", rb"...", ...) need no
// special handling: the prefix letters aren't quote characters, so they pass
// through as ordinary text and the following quote starts string-tracking
// exactly as it would unprefixed.
func stripPythonComments(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	n := len(text)
	i := 0
	for i < n {
		c := text[i]
		if c == '"' || c == '\'' {
			quote := c
			if i+2 < n && text[i+1] == quote && text[i+2] == quote {
				// Triple-quoted string: copy verbatim up to and including
				// the matching closing triple-quote (or to EOF if it's
				// unterminated — malformed input, just stop tracking).
				b.WriteString(text[i : i+3])
				i += 3
				closer := text[i-3 : i]
				if idx := strings.Index(text[i:], closer); idx == -1 {
					b.WriteString(text[i:])
					i = n
				} else {
					b.WriteString(text[i : i+idx+3])
					i += idx + 3
				}
				continue
			}
			// Single-line string: copy verbatim, honoring backslash
			// escapes (including an escaped newline, Python's explicit
			// line-continuation inside such a string), up to the matching
			// closing quote. An unescaped newline before the closing quote
			// means the source is malformed (or this is actually a
			// comment's "#" miscategorized as a quote, which can't happen
			// since we only enter this branch on a real quote byte) — stop
			// string-tracking at the newline rather than consuming past it.
			b.WriteByte(c)
			i++
			for i < n {
				if text[i] == '\\' && i+1 < n {
					b.WriteByte(text[i])
					b.WriteByte(text[i+1])
					i += 2
					continue
				}
				b.WriteByte(text[i])
				done := text[i] == quote || text[i] == '\n'
				i++
				if done {
					break
				}
			}
			continue
		}
		if c == '#' {
			for i < n && text[i] != '\n' {
				i++
			}
			continue
		}
		b.WriteByte(c)
		i++
	}
	return b.String()
}

func isSourceFile(path string) bool {
	for _, ext := range []string{".py", ".ipynb"} {
		if strings.HasSuffix(path, ext) {
			return true
		}
	}
	return false
}
