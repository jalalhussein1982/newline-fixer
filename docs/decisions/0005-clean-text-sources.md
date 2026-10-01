# 0005. Clean text sources: Wikipedia dump plus generated documents; Wikitext rejected

Date: 2026-10-01. Status: accepted.

## Context

The training data needs clean English documents with headings, lists and paragraphs
whose whitespace can serve as the reference. Design v1 listed Wikitext-103 raw, the
Wikipedia dump and LLM-generated documents.

## Options

1. **Wikitext-103 raw.** Convenient, but its "raw" variant is still tokenized: spaces
   before commas and periods, `'s` split off, `@-@` for hyphens. Its whitespace is not
   clean reference formatting. Using it would teach every model to put spaces before
   punctuation. It is also derived from Wikipedia, so it overlaps the dump.
2. **Wikipedia dump via `wikimedia/wikipedia` on the Hub, pinned revision.** Real
   article text with paragraphs and section titles, article ids for grouping, proper
   spacing. Few bullet lists.
3. **LLM-generated documents.** The only source where list and heading variety can be
   demanded. Not regenerable identically, so the generated set must be kept and
   published.
4. **A Markdown documentation corpus.** Rich in lists and headings; needs Markdown
   stripping and licence checks. Deferred.

## Decision

Options 2 and 3, with 4 deferred until first results show whether list coverage is
sufficient. Option 1 is rejected.

## Consequences

- One real source and one generated source. The report discusses the generated share
  and its risks.
- Article ids are stored and used to group documents before splitting.
- The built dataset, including every generated document, is published with content
  hashes and a version, so results are reproducible from the artifact.
