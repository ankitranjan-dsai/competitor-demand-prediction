# Task 03 — Text pipeline validation (evidence)

_Generated 2026-07-25 by `src/validate_text_pipeline.py`._

The Google backfill has no `job_description` text, so the Layer B
text pipeline is validated here against **real job-posting HTML**
from the Arbeitnow public API (no auth). This proves the cleaner and
tokeniser work on genuine posting prose before the Adzuna pull lands.
None of this data enters the Google dataset.

**What the pipeline must survive:** HTML tags and framework attributes
(`<div data-testid=...>`), HTML entities (`&amp;`, `&nbsp;`), block
tags that must not glue words together, accented characters, and
technical tokens that must stay intact.

## Sample 1 — Engineering Manager - Edge AI (SumUp)

- raw length: **6,301** chars · cleaned: **4,297** chars (**32%** markup removed)
- tokens after stopword removal: **366** (291 unique)

**Raw (first 300 chars):**

```html
<p>Our AI Platform &amp; Edge team sits at the heart of how SumUp is scaling intelligent, merchant-facing experiences across millions of interactions. We build and operate the AI systems — from LLM-powered assistants to agentic platforms — that directly shape how merchants get support, resolve issue
```

**Cleaned (first 300 chars):**

```text
Our AI Platform & Edge team sits at the heart of how SumUp is scaling intelligent, merchant-facing experiences across millions of interactions. We build and operate the AI systems - from LLM-powered assistants to agentic platforms - that directly shape how merchants get support, resolve issues, and 
```

**Top tokens:** ai (9), sumup (9), across (5), businesses (4), edge (3), llm-powered (3), products (3), engineering (3), lead (3), delivery (3), technical (3), environment (3)

## Sample 2 — Senior Full-Stack Engineer - SuperApp (SumUp)

- raw length: **6,649** chars · cleaned: **3,919** chars (**41%** markup removed)
- tokens after stopword removal: **335** (264 unique)

**Raw (first 300 chars):**

```html
<div data-testid="markdown-response">
<p class="p1">The SuperApp squad owns parts of the merchant experience across Web, Android, iOS and the flows that millions of small business owners use every day to run and grow their businesses. As our dedicated full stack web engineer within the squad, you'll
```

**Cleaned (first 300 chars):**

```text
The SuperApp squad owns parts of the merchant experience across Web, Android, iOS and the flows that millions of small business owners use every day to run and grow their businesses. As our dedicated full stack web engineer within the squad, you'll own this space entirely, shaping technical directio
```

**Top tokens:** sumup (9), web (6), businesses (5), across (4), technical (4), millions (3), small (3), product (3), strong (3), environment (3), squad (2), merchant (2)

## Sample 3 — Senior Data Science/ML Engineer - Financial Crime (SumUp)

- raw length: **8,219** chars · cleaned: **5,694** chars (**31%** markup removed)
- tokens after stopword removal: **496** (345 unique)

**Raw (first 300 chars):**

```html
<div data-testid="markdown-response">
<h3><strong>Team description</strong></h3>
<p>The Risk AI Engineering Squad is a cross-functional team within the Risk &amp; Compliance tribe,&nbsp; responsible for developing cutting-edge data products and ML solutions that power Transaction Monitoring and prot
```

**Cleaned (first 300 chars):**

```text
Team description The Risk AI Engineering Squad is a cross-functional team within the Risk & Compliance tribe, responsible for developing cutting-edge data products and ML solutions that power Transaction Monitoring and protect our merchant base from Money Laundering and Financial Crime. We combine a
```

**Top tokens:** ml (9), monitoring (9), data (7), sumup (7), risk (6), engineering (6), aml (6), model (6), models (5), compliance (4), products (4), transaction (4)

## Automated checks

- PASS — no HTML tags remain
- PASS — no unescaped entities
- PASS — no non-breaking spaces
- PASS — text is non-empty
- PASS — ascii-normalised
