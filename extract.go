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

// pipelineCallPattern matches pipeline(...)'s whole argument list as one
// blob, the same whole-call-first approach hfHubDownloadCallPattern uses for
// hf_hub_download/snapshot_download, so the model ID — whether a quoted
// literal or a same-file constant name (see resolveTokenID) — can be found
// regardless of whether it's passed as pipeline's second positional
// argument or its model= keyword.
var pipelineCallPattern = regexp.MustCompile(`\bpipeline\(([^)]*)\)`)

// pipelineTaskPrefixPattern matches pipeline's first positional argument (a
// quoted task name, e.g. "text-classification") plus the comma separating
// it from a second positional argument, if one follows — the model ID,
// when passed positionally rather than as model=, is that second argument,
// immediately after this prefix.
var pipelineTaskPrefixPattern = regexp.MustCompile(`^\s*["'][^"']*["']\s*,\s*`)

// modelKeywordPrefixPattern matches pipeline's model= keyword up to (but
// not including) the value that follows it, wherever model= appears in the
// call's argument list.
var modelKeywordPrefixPattern = regexp.MustCompile(`\bmodel\s*=\s*`)

// loadDatasetCallPattern matches load_dataset(...)'s whole argument list,
// the same whole-call-first approach as pipelineCallPattern and
// hfHubDownloadCallPattern, so its first positional argument — the dataset
// ID — can be resolved through resolveTokenID exactly like the others,
// rather than requiring a quoted literal directly after the open paren.
var loadDatasetCallPattern = regexp.MustCompile(`\bload_dataset\(([^)]*)\)`)

// fromPretrainedCallPattern captures the whole argument list of a
// `<something>.from_pretrained(...)` call — covers AutoModel, AutoTokenizer,
// every task-specific Auto* class, and third-party wrappers
// (SentenceTransformer.from_pretrained, CLIPModel, ...) — the same way
// hfHubDownloadCallPattern does for hf_hub_download/snapshot_download.
// Capturing the whole call first, rather than matching only the first
// quoted argument directly (this pattern's previous shape), is needed
// because a handful of from_pretrained-style classmethods accept more than
// one Hub ID as separate leading positional arguments: the real setfit
// library's AbsaModel.from_pretrained(model_id, polarity_model_id=None,
// ...) takes two independent model IDs positionally (confirmed live against
// huggingface/setfit's own tests/conftest.py, which calls it with both as
// plain positional string literals) — see leadingPositionalIDs.
var fromPretrainedCallPattern = regexp.MustCompile(`\.from_pretrained\(([^)]*)\)`)

// leadingPositionalIDPattern matches one leading positional argument that's
// a bare Hub-ID string literal — nothing else in its comma-delimited slot —
// plus the comma separating it from the next positional argument, if one
// follows. leadingPositionalIDs uses this to walk a call's argument list
// collecting every ID passed as its own separate positional argument.
var leadingPositionalIDPattern = regexp.MustCompile(`^\s*["'](` + idPattern + `)["']\s*(,)?`)

// leadingPositionalIDs returns every Hub-ID string literal passed as a
// separate leading positional argument in a call's argument-list text
// (everything between its parens), resolving a bare name in that position
// through constants (see collectStringConstants) when it's a simple
// same-file literal assignment. Python requires positional arguments to
// precede any keyword argument in a call, so once a comma-delimited slot
// fails to be a bare quoted ID or a resolvable name — because it's a
// keyword argument, an unresolvable variable, or any other expression —
// nothing after it can be a positional ID either, and the walk stops there.
// Almost every from_pretrained call has exactly one such argument (the
// common case this returns a single-element slice for), but see
// fromPretrainedCallPattern's doc comment for the real multi-ID exception
// this generalizes to.
func leadingPositionalIDs(args string, constants map[string]string) []string {
	var ids []string
	pos := 0
	for {
		rest := args[pos:]
		if m := leadingPositionalIDPattern.FindStringSubmatchIndex(rest); m != nil {
			ids = append(ids, rest[m[2]:m[3]])
			if m[4] == -1 {
				// No trailing comma matched: this was the last (or only)
				// positional argument in the list.
				break
			}
			pos += m[1]
			continue
		}

		im := bareIdentifierPattern.FindStringSubmatchIndex(rest)
		if im == nil {
			break
		}
		name := rest[im[2]:im[3]]
		after := rest[im[1]:]
		trimmed := strings.TrimLeft(after, " \t\n")
		if strings.HasPrefix(trimmed, "=") && !strings.HasPrefix(trimmed, "==") {
			// name=value: this is a keyword argument, not a bare positional
			// name — the same "nothing after can be positional" rule as any
			// other non-ID expression applies.
			break
		}
		id, ok := constants[name]
		if !ok {
			// An unresolvable name (imported, computed, or just not a
			// simple same-file literal) — can't be a Hub ID this tool can
			// check, exactly like any other non-literal positional
			// argument.
			break
		}
		ids = append(ids, id)
		if strings.HasPrefix(trimmed, ",") {
			pos += im[1] + (len(after) - len(trimmed)) + 1
			continue
		}
		break
	}
	return ids
}

// stringConstantPattern matches a simple same-file assignment of a Hub-ID
// string literal to a bare name, alone on its line: NAME = "org/name", or
// type-annotated, NAME: str = "org/name". Real training/eval scripts
// routinely pull a Hub ID out into a named constant near the top of the
// file and pass the name to from_pretrained(...) rather than the literal
// itself — confirmed live in accelerate's own tests/fsdp/test_fsdp.py:
//
//	LLAMA_TESTING = "hf-internal-testing/tiny-random-LlamaForCausalLM"
//	...
//	model = AutoModel.from_pretrained(LLAMA_TESTING)
//
// Before collectStringConstants/leadingPositionalIDs' use of it, a bare name
// standing in for the ID argument was indistinguishable from any other
// non-literal expression (a config lookup, a CLI argument, ...) and the
// call was silently skipped — a hallucinated or typosquatted ID assigned to
// a constant and referenced this way produced zero findings and exit 0,
// invisible exactly like the gaps closed in previous passes.
//
// This resolves only the simple, unambiguous one-hop case: a name defined
// by exactly one literal assignment statement, by itself on its line.
// Anything built from an f-string, concatenation, a function call, or
// imported from another module is deliberately left unresolved, the same
// as any other non-literal expression — correctly invisible rather than
// guessed at. Matching is purely textual, not scope-aware: a name reused
// with different values in different functions resolves to whichever
// assignment appears last in the file, a known imprecision acceptable in a
// regex-based tool that was never a full Python parser.
var stringConstantPattern = regexp.MustCompile(`(?m)^[ \t]*([A-Za-z_][A-Za-z0-9_]*)[ \t]*(?::[^=\n]+)?=[ \t]*["'](` + idPattern + `)["'][ \t]*$`)

// collectStringConstants returns every name->ID mapping stringConstantPattern
// finds in text. Called once per extractReferences invocation (text is
// already comment/docstring-stripped by then) and threaded through to every
// extraction path that resolves a bare identifier positional argument.
func collectStringConstants(text string) map[string]string {
	consts := map[string]string{}
	for _, m := range stringConstantPattern.FindAllStringSubmatch(text, -1) {
		consts[m[1]] = m[2]
	}
	return consts
}

// bareIdentifierPattern matches a single leading Python identifier —
// leadingPositionalIDs uses this to recognize a positional argument that's
// a bare name rather than a quoted literal, so it can try resolving the
// name through collectStringConstants' map instead of giving up on it.
var bareIdentifierPattern = regexp.MustCompile(`^\s*([A-Za-z_][A-Za-z0-9_]*)`)

// resolveTokenID extracts a Hub-ID value from the start of s: either a
// quoted literal (via positionalRepoIDArgPattern), or a bare identifier
// resolved through constants (see collectStringConstants) — the same two
// shapes leadingPositionalIDs already accepts for from_pretrained's
// positional arguments, generalized here for every other single-argument
// position that can equally hold either shape in real code: pipeline's
// model= keyword and second positional argument, load_dataset's first
// positional argument, and hf_hub_download/snapshot_download's repo_id=
// keyword and positional argument. Before this existed, a Hub ID assigned
// to a same-file constant and passed to any of those four call shapes
// (rather than from_pretrained, the one shape leadingPositionalIDs already
// covered) was invisible to extraction — confirmed live against
// huggingface/datasets' own tests/test_load.py:
// `SAMPLE_DATASET_IDENTIFIER3 = "hf-internal-testing/multi_dir_dataset"`
// then `load_dataset(SAMPLE_DATASET_IDENTIFIER3)` produced zero findings.
// Returns ok=false if s starts with neither shape — a keyword argument's
// own name (NAME=value), an unresolvable name, or any other non-ID
// expression — exactly the same "correctly invisible rather than guessed
// at" rule leadingPositionalIDs already applies.
func resolveTokenID(s string, constants map[string]string) (string, bool) {
	if m := positionalRepoIDArgPattern.FindStringSubmatch(s); m != nil {
		return m[1], true
	}
	im := bareIdentifierPattern.FindStringSubmatchIndex(s)
	if im == nil {
		return "", false
	}
	name := s[im[2]:im[3]]
	rest := strings.TrimLeft(s[im[1]:], " \t\n")
	if strings.HasPrefix(rest, "=") && !strings.HasPrefix(rest, "==") {
		// name=value: this is a keyword argument, not a bare identifier
		// value — the same "nothing here can be a positional/value ID"
		// rule leadingPositionalIDs applies to the exact same shape.
		return "", false
	}
	id, ok := constants[name]
	return id, ok
}

// hfHubDownloadCallPattern matches an hf_hub_download(...)/snapshot_download(...)
// call's whole argument list as one blob, rather than jumping straight to
// repo_id=, the same whole-call-first approach pipelineCallPattern and
// loadDatasetCallPattern use. Both functions also
// take a repo_type= keyword telling the Hub whether repo_id names a model, a
// dataset, or a Space, and repo_type can legally appear either before or
// after repo_id since both are keyword arguments — capturing the whole call
// first lets repoTypeArgPattern find it regardless of order.
var hfHubDownloadCallPattern = regexp.MustCompile(`\b(?:hf_hub_download|snapshot_download)\(([^)]*)\)`)

// repoIDKeywordPrefixPattern matches the repo_id= keyword up to (but not
// including) the value that follows it, wherever repo_id= appears in the
// call's argument list. resolveTokenID reads the value from there, so this
// covers both a quoted literal (repoIDArgPattern's old job) and a same-file
// constant name in one place.
var repoIDKeywordPrefixPattern = regexp.MustCompile(`\brepo_id\s*=\s*`)

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
// back a false "not_found" hallucination — and the same is true of
// repo_type="space": confirmed live against real conversion/test scripts in
// huggingface/transformers' own source (not docstrings — e.g.
// models/owlv2/convert_owlv2_to_hf.py's
// `hf_hub_download(repo_id="adirik/OWL-ViT", repo_type="space",
// filename="assets/astronaut.png")`), every one of those real, existing
// Spaces returns 401 from /api/models/ (not_found, by this tool's own
// mapping) and 200 from /api/spaces/.
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
// Python "#" comments and triple-quoted docstring bodies are stripped first
// (see stripDeadPythonText). Callers pass .ipynb files through
// decodeNotebookSource first (see notebook.go), so by the time text reaches
// here it's always plain decoded Python source — real code cell text, not
// the notebook's own raw JSON — and stripping applies to it exactly as it
// does to an ordinary .py file. Commented-out calls are routine in real ML code — e.g. diffusers' own
// pipelines/stable_diffusion/convert_from_ckpt.py keeps a superseded
// from_pretrained call around as a comment while iterating on the
// replacement — and the whole point of this tool is flagging IDs the real
// program will actually try to fetch, not dead code. Without this, a typo
// or hallucinated name left behind in a commented-out call is reported
// exactly like a live one, a false positive this tool has no business
// raising.
//
// The same reasoning applies to a docstring's "Example:" block: confirmed
// live against transformers' own source, dozens of its per-model
// modeling_*.py files carry a docstring Example showing
// `FooForCausalLM.from_pretrained("meta-foo/Foo-2-7b-hf")` — a
// cookiecutter-template placeholder checkpoint name that was never a real
// Hub repo and, per the real API, 401s like any other nonexistent one. That
// text sits inside a triple-quoted string; it documents how the class would
// be used, but it's never executed as part of the program the docstring
// belongs to, exactly like a commented-out call. Without stripping it, a
// single popular library's placeholder-naming convention floods every
// finding list with not_found noise and fails any build using hfaudit's
// default -fail-on.
func extractReferences(text, label string) []repoRef {
	text = stripDeadPythonText(text)
	constants := collectStringConstants(text)
	var refs []repoRef
	for _, m := range pipelineCallPattern.FindAllStringSubmatchIndex(text, -1) {
		args := text[m[2]:m[3]]
		line := 1 + strings.Count(text[:m[0]], "\n")
		if pm := pipelineTaskPrefixPattern.FindStringIndex(args); pm != nil {
			if id, ok := resolveTokenID(args[pm[1]:], constants); ok {
				refs = append(refs, repoRef{id: id, kind: kindModel, source: fmt.Sprintf("%s:%d", label, line)})
			}
		}
		if km := modelKeywordPrefixPattern.FindStringIndex(args); km != nil {
			if id, ok := resolveTokenID(args[km[1]:], constants); ok {
				refs = append(refs, repoRef{id: id, kind: kindModel, source: fmt.Sprintf("%s:%d", label, line)})
			}
		}
	}
	for _, m := range loadDatasetCallPattern.FindAllStringSubmatchIndex(text, -1) {
		args := text[m[2]:m[3]]
		id, ok := resolveTokenID(args, constants)
		if !ok {
			continue
		}
		line := 1 + strings.Count(text[:m[0]], "\n")
		refs = append(refs, repoRef{id: id, kind: kindDataset, source: fmt.Sprintf("%s:%d", label, line)})
	}
	for _, m := range fromPretrainedCallPattern.FindAllStringSubmatchIndex(text, -1) {
		args := text[m[2]:m[3]]
		line := 1 + strings.Count(text[:m[0]], "\n")
		for _, id := range leadingPositionalIDs(args, constants) {
			refs = append(refs, repoRef{
				id:     id,
				kind:   kindModel,
				source: fmt.Sprintf("%s:%d", label, line),
			})
		}
	}
	for _, m := range hfHubDownloadCallPattern.FindAllStringSubmatchIndex(text, -1) {
		args := text[m[2]:m[3]]
		var id string
		var ok bool
		if km := repoIDKeywordPrefixPattern.FindStringIndex(args); km != nil {
			id, ok = resolveTokenID(args[km[1]:], constants)
		}
		if !ok {
			id, ok = resolveTokenID(args, constants)
		}
		if !ok {
			continue
		}
		kind := kindModel
		if tm := repoTypeArgPattern.FindStringSubmatch(args); tm != nil {
			switch tm[1] {
			case "dataset":
				kind = kindDataset
			case "space":
				kind = kindSpace
			}
		}
		line := 1 + strings.Count(text[:m[0]], "\n")
		refs = append(refs, repoRef{
			id:     id,
			kind:   kind,
			source: fmt.Sprintf("%s:%d", label, line),
		})
	}
	return refs
}

// stripDeadPythonText removes Python "#"-to-end-of-line comments and
// triple-quoted string bodies (overwhelmingly docstrings — see below) from
// text, while leaving ordinary single/double-quoted string literal contents
// (including anything that happens to contain a "#", like an f-string or a
// URL) untouched, and without disturbing line numbers: removed bytes are
// replaced one-for-one with spaces (not deleted outright, except a "#"
// comment's own bytes which are dropped since nothing after them on the
// line matters), and every newline byte is kept as-is, so
// 1+strings.Count(text[:i], "\n") against the result still lines up with the
// original source.
//
// Triple-quoted strings are blanked rather than left verbatim like other
// string literals because, unlike a short quoted argument such as
// "facebook/bart-large", they are essentially always a function/class/
// module docstring, and real-world docstrings routinely embed a markdown or
// reStructuredText "Example:" code block showing illustrative, non-executed
// calls. Confirmed live against transformers' own source: dozens of its
// per-model modeling_*.py files carry a docstring Example with
// `FooForCausalLM.from_pretrained("meta-foo/Foo-2-7b-hf")` — a
// cookiecutter-template placeholder name, not a real Hub repo (401s on the
// real API like any other nonexistent one) — left over from the model's own
// code-generation template. That text is never executed as part of the
// program the docstring documents, exactly like a "#" comment, so it must
// not be extracted and checked alongside genuinely live calls.
//
// This is a small hand-rolled Python lexer, not a full one — it only needs
// to track enough state (single/double/triple-quoted strings, backslash
// escapes) to tell a real comment-starting "#" apart from one sitting inside
// a string. String prefixes (r"...", f"...", b"...", rb"...", ...) need no
// special handling: the prefix letters aren't quote characters, so they pass
// through as ordinary text and the following quote starts string-tracking
// exactly as it would unprefixed.
func stripDeadPythonText(text string) string {
	var b strings.Builder
	b.Grow(len(text))
	n := len(text)
	i := 0
	for i < n {
		c := text[i]
		if c == '"' || c == '\'' {
			quote := c
			if i+2 < n && text[i+1] == quote && text[i+2] == quote {
				// Triple-quoted string: blank the whole span (including its
				// own quote markers) up to and including the matching
				// closing triple-quote (or to EOF if it's unterminated —
				// malformed input, just stop tracking), keeping newlines so
				// line numbers for anything after it stay accurate.
				opener := text[i : i+3]
				end := n
				if idx := strings.Index(text[i+3:], opener); idx != -1 {
					end = i + 3 + idx + 3
				}
				for ; i < end; i++ {
					if text[i] == '\n' {
						b.WriteByte('\n')
					} else {
						b.WriteByte(' ')
					}
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
