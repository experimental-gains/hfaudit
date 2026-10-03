# hfaudit

Catch hallucinated or typosquatted Hugging Face model/dataset IDs
before you run `from_pretrained(...)` on one.

LLM coding assistants invent plausible-looking package names that
don't exist — "package hallucination" — and the same thing happens
with Hugging Face Hub IDs: an assistant suggests `AutoModel.
from_pretrained("meta-llama/Llama-3-fake-variant")` and there's no
such repo. Existing Hub-security tools (`modelscan`, `picklescan`,
`ModelAudit`) all scan a model's *file contents* for unsafe
deserialization once you've already downloaded it. None of them check
whether the *name* itself is real, or whether it's suspiciously close
to a well-known org — the same gap `slopcheck` closes for PyPI/npm.

hfaudit scans Python source for Hub references and checks each one
against the Hub's public, unauthenticated metadata API:

- **not_found** — no such repo. Either a typo, or a name an LLM
  hallucinated outright. Worth a hard look before running the code.
- **ok** — the repo exists. If it's gated (requires accepting terms
  to download files), that's reported too, but doesn't fail the
  build by default — being gated isn't itself suspicious.
- **error** — the lookup itself didn't complete (network failure,
  rate limit). Doesn't fail the build by default.

Every `ok` and `not_found` finding is also checked against a list of
well-known Hub orgs (`openai`, `google`, `meta-llama`, `microsoft`,
`stabilityai`, ...): if the namespace is one or two edits away from
one of those — `0penai` instead of `openai`, say — it's flagged as a
possible typosquat/impersonation, regardless of whether the repo
itself exists (an attacker can both register a look-alike name *and*
get it indexed).

## Install

```
go install github.com/experimental-gains/hfaudit@latest
```

## Usage

```
hfaudit .                      # scan every .py/.ipynb file under the current directory
hfaudit path/to/file.py        # scan one file
hfaudit -id openai/clip-vit-base-patch32   # check an explicit ID, no scanning
cat script.py | hfaudit        # read source from stdin
```

Recognized call shapes: `X.from_pretrained("org/name")` (any class —
`AutoModel`, `AutoTokenizer`, `SentenceTransformer`, ...),
`pipeline("task", "org/name")` and `pipeline(..., model="org/name")`,
`load_dataset("org/name")`, `hf_hub_download(repo_id="org/name")`,
`snapshot_download(repo_id="org/name")`. IDs with no `/` (e.g.
`"bert-base-uncased"` with no namespace) or more than one `/` (a local
path) aren't Hub IDs this tool can look up, and are skipped.

```
$ hfaudit .
not_found model   0penai/this-definitely-does-not-exist-xyz  [looks like "openai", edit distance 1]  (model.py:4)
ok        model   openai/clip-vit-base-patch32  (model.py:3)
ok        dataset stanfordnlp/imdb  (model.py:7)
```

Exit code is `1` if any finding matches `-fail-on` (default
`not_found,typosquat`), `0` otherwise. Pass `-json` for machine-
readable output, or `-fail-on none` to only report, never fail.

```
-fail-on not_found,typosquat,gated,error   # any comma-separated subset, or "none"
```

## Why an unauthenticated API call is enough

The Hub's `/api/models/{id}` and `/api/datasets/{id}` endpoints
answer anonymously: a real public repo returns full metadata with
HTTP 200 — including gated ones, like `meta-llama/Llama-2-7b`, since
gating restricts downloading files, not reading metadata. A missing
namespace, a missing repo name under a real namespace, and (almost
certainly) a private repo you can't see all answer identically with
HTTP 401 and `{"error":"Invalid username or password."}` — an
auth-shaped error that in practice just means "nothing here,
anonymously." There's no separate 404 for this endpoint. hfaudit
treats that 401 as `not_found` and reads the `gated` field out of the
200 body instead of inferring it from the HTTP status.

## CI

```yaml
- uses: experimental-gains/hfaudit@main
  with:
    args: .
```

## Related tools

Other no-signup CLIs from the same org:

- **[slopcheck](https://github.com/experimental-gains/slopcheck)** — the same hallucinated/typosquatted-name check for PyPI/npm dependency names
- **[modslop](https://github.com/experimental-gains/modslop)** — the same check for Go module paths in `go.mod`
- **[goproxycheck](https://github.com/experimental-gains/goproxycheck)** — diagnoses why a Go module version won't fetch via the public proxy/sumdb
- **[goprivaudit](https://github.com/experimental-gains/goprivaudit)** — audits `GOPRIVATE`/`GONOSUMDB` config for private-module sumdb leaks

## Support

If this caught something useful, a star helps others find it — that's
the main thing. This project is free and open source; if it's useful
to you, tips are also welcome via
[Liberapay](https://liberapay.com/experimental-gains/) or this ETH
address (self-custody, no KYC, no obligation):
`0x87053a1898994043e7476800cB5d4BDB423eADD7`

Build-in-public updates on [Nostr](https://njump.me/npub19ycp547pcykycy9kw3y04fe0wn3uukdukdhcdjdjce5s5ueg4qwq6un59y) (no account needed to read).

## License

MIT
