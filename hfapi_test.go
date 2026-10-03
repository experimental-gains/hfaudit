package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func newTestHFClient(t *testing.T, handler http.HandlerFunc) *httpHFClient {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return &httpHFClient{http: srv.Client(), base: srv.URL}
}

func TestCheckSendsNoAuthByDefault(t *testing.T) {
	c := newTestHFClient(t, func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("Authorization header = %q, want empty with no token env vars set", auth)
		}
		w.WriteHeader(http.StatusUnauthorized)
	})
	if got := c.check("acme/private-model", kindModel).status; got != "not_found" {
		t.Errorf("status = %q, want not_found", got)
	}
}

func TestCheckSendsHFToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_realtoken")
	c := newTestHFClient(t, func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Header.Get("Authorization"), "Bearer hf_realtoken"; got != want {
			t.Errorf("Authorization header = %q, want %q", got, want)
		}
		w.Write([]byte(`{"gated":false}`))
	})
	res := c.check("acme/private-model", kindModel)
	if res.status != "ok" {
		t.Errorf("status = %q, want ok (the whole point: a private repo the token can see)", res.status)
	}
}

func TestCheckFallsBackToLegacyTokenName(t *testing.T) {
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacytoken")
	c := newTestHFClient(t, func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Header.Get("Authorization"), "Bearer hf_legacytoken"; got != want {
			t.Errorf("Authorization header = %q, want %q", got, want)
		}
		w.Write([]byte(`{"gated":false}`))
	})
	if got := c.check("acme/private-model", kindModel).status; got != "ok" {
		t.Errorf("status = %q, want ok", got)
	}
}

func TestCheckHFTokenTakesPriorityOverLegacyName(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_new")
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "hf_legacy")
	c := newTestHFClient(t, func(w http.ResponseWriter, r *http.Request) {
		if got, want := r.Header.Get("Authorization"), "Bearer hf_new"; got != want {
			t.Errorf("Authorization header = %q, want %q (HF_TOKEN must win, matching huggingface_hub's own precedence)", got, want)
		}
		w.Write([]byte(`{"gated":false}`))
	})
	c.check("acme/private-model", kindModel)
}

func TestCheckRespectsDisableImplicitToken(t *testing.T) {
	t.Setenv("HF_TOKEN", "hf_realtoken")
	t.Setenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
	c := newTestHFClient(t, func(w http.ResponseWriter, r *http.Request) {
		if auth := r.Header.Get("Authorization"); auth != "" {
			t.Errorf("Authorization header = %q, want empty: HF_HUB_DISABLE_IMPLICIT_TOKEN means the real huggingface_hub call wouldn't send it either", auth)
		}
		w.WriteHeader(http.StatusUnauthorized)
	})
	if got := c.check("acme/private-model", kindModel).status; got != "not_found" {
		t.Errorf("status = %q, want not_found", got)
	}
}
