package main

import (
	"encoding/json"
	"net/http"
	"os"
	"time"
)

// hfClient checks whether a Hugging Face Hub repo ID exists, using the
// Hub's public metadata API. No account or token is required to use this
// tool — but if the real huggingface_hub-based code being audited would
// itself authenticate (because HF_TOKEN/HUGGING_FACE_HUB_TOKEN is set in
// the same environment hfaudit runs in, exactly as the real library reads
// it), this client authenticates too, so a legitimate private repo
// reference isn't misreported as "not_found" alongside genuine typos.
type hfClient interface {
	check(id string, kind repoKind) checkResult
}

type checkResult struct {
	status string // "ok", "not_found", or "error"
	gated  string // "", "auto", or "manual" — only meaningful when status is "ok"
}

type httpHFClient struct {
	http *http.Client
	base string
}

func defaultHFClient() hfClient {
	return &httpHFClient{
		http: &http.Client{Timeout: 10 * time.Second},
		base: "https://huggingface.co",
	}
}

// hfRepoMeta is the subset of the Hub API's response this tool reads.
// "gated" is "auto", "manual", or false (a bool, not a string) when the
// repo isn't gated at all — json.RawMessage defers decoding so the bool
// case doesn't fail to unmarshal into a string field.
type hfRepoMeta struct {
	Gated json.RawMessage `json:"gated"`
}

// check returns "ok" for a repo that exists (gated or not — gating only
// restricts downloading files, not reading metadata), "not_found" when it
// doesn't, and "error" when the lookup itself didn't complete.
//
// Confirmed live against the real API (no auth): a missing namespace, a
// real namespace with a nonexistent repo name, and a private repo an
// anonymous caller can't see all answer identically with HTTP 401 and body
// `{"error":"Invalid username or password."}` — an auth-shaped error that
// in practice just means "nothing here, anonymously". There is no separate
// 404 for this endpoint. Gating is reported inside the 200 body's "gated"
// field instead (seen "manual" on meta-llama/Llama-2-7b, which still
// returns full metadata with no auth).
//
// A private repo only stops looking identical to a nonexistent one once
// the request authenticates as someone with access — confirmed from
// huggingface_hub's own source (utils/_auth.py's get_token(),
// utils/_headers.go's build_hf_headers(): "by default, authorization token
// is always provided... retrieved from the cache (implicit use)") and the
// Hub docs ("you must be authenticated... to download private repos"). So
// the real from_pretrained()/hf_hub_download()/load_dataset() call this
// tool's findings describe already sends HF_TOKEN (falling back to the
// legacy HUGGING_FACE_HUB_TOKEN name) as `Authorization: Bearer` on every
// request by default — unless HF_HUB_DISABLE_IMPLICIT_TOKEN disables that.
// Mirroring the same env vars here means hfaudit, run in the same
// environment as the real code, reports what that code would actually see
// instead of flattening "private, and I can get at it" and "doesn't exist"
// into the same not_found finding.
func (c *httpHFClient) check(id string, kind repoKind) checkResult {
	endpoint := "models"
	if kind == kindDataset {
		endpoint = "datasets"
	}
	// id's own "/" separates the org and name segments and must stay a
	// literal path separator, so build the URL by concatenation rather
	// than url.PathEscape(id), which would percent-encode it too.
	u := c.base + "/api/" + endpoint + "/" + id

	req, err := http.NewRequest(http.MethodGet, u, nil)
	if err != nil {
		return checkResult{status: "error"}
	}
	if os.Getenv("HF_HUB_DISABLE_IMPLICIT_TOKEN") == "" {
		if token := os.Getenv("HF_TOKEN"); token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		} else if token := os.Getenv("HUGGING_FACE_HUB_TOKEN"); token != "" {
			req.Header.Set("Authorization", "Bearer "+token)
		}
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return checkResult{status: "error"}
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		var meta hfRepoMeta
		gated := ""
		if json.NewDecoder(resp.Body).Decode(&meta) == nil {
			var s string
			if json.Unmarshal(meta.Gated, &s) == nil {
				gated = s
			}
		}
		return checkResult{status: "ok", gated: gated}
	case http.StatusUnauthorized:
		return checkResult{status: "not_found"}
	default:
		return checkResult{status: "error"}
	}
}
