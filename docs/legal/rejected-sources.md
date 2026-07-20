# Rejected sources — provenance failures

Sources rejected under the Task 01 rules even though they are technically easy
to access. Kept on record so the team doesn't re-litigate them later.

## LinkedIn-derived datasets (Hugging Face / Kaggle re-uploads)

Several datasets circulate with permissive-looking tags (e.g.
`xanderios/linkedin-job-postings`, tagged MIT). **Rejected:** the underlying
data was scraped from LinkedIn, whose User Agreement prohibits scraping. A
re-uploader cannot grant a licence to data they had no right to collect —
the licence tag does not cure the provenance. Same reasoning applies to any
Indeed- or Glassdoor-scraped dataset.

## Remotive API

Legally fine (public API with a stated sharing-oriented notice), but its legal
notice restricts re-posting jobs to third-party sites, and coverage is
remote-first companies — checked 2026-07-20, no Google postings. Rejected for
the Google track on **coverage**, not legality; other specialists may still
find it useful.

## Arbeitnow API

Public, no-auth API; checked 2026-07-20 — coverage is EU/German SMEs, no
Google-family employers on sampled pages. Rejected for the Google track on
coverage.
