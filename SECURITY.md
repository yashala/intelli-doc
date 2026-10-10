# Security

## Scope and deployment assumptions
IntelliDoc is a single-user, self-hosted service intended to run on a workstation through
Docker Compose. It has **no authentication**. Ports are bound to `127.0.0.1`. Do not expose
it to a network without adding authentication and the controls listed in
`docs/limitations.md`.

## What is protected
- Model credentials live only in the API container environment (`.env`, never committed).
  The browser bundle contains no keys.
- Uploads are limited to 25 MB and must start with the PDF signature.
- Model output is rendered as plain text in the UI.
- Citations are resolved server-side from retrieved chunk ids; the model cannot invent page
  references.
- Text in documents that looks like instructions to an AI is flagged and passed to the model as
  quoted data. This reduces but does not prevent prompt injection; the benchmark reports how
  often injected instructions were followed.

## Known gaps
- PDF parsing runs in the API process with no sandbox, CPU or memory limit.
- Retrieved document text is sent to the configured model provider when `LLM_MODE` is `openai` or `gemini`.
- No rate limiting.

## Reporting
Please report suspected vulnerabilities privately through GitHub's
"Report a vulnerability" (security advisories) on this repository rather than a public issue.
