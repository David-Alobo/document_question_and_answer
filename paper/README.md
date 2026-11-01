# Manuscript

LaTeX source of the paper and its SI (arXiv preprint first).

```bash
uv run python paper/make_numbers.py     # numbers, tables and prompts from eval/results/
python paper/make_bib.py                # references.bib from DOIs (only when references change)
bash paper/build.sh                     # in WSL or Linux: builds build/main.pdf and build/si.pdf
```

- `\result{...}` (orange) marks numbers still to come from running experiments; `\todo{...}`
  (red) marks other gaps. Search for both before submitting.
- Every number in the text is a macro from `generated/numbers.tex`, and every data table a file
  in `generated/`, written by `make_numbers.py` from `eval/results/`. Edit the script, never
  its output.
- Every reference is fetched from its DOI (Crossref or DataCite) by `make_bib.py`. The one
  without a DOI (Holm 1979) is listed in `MANUAL` with the record it was checked against.
- The app's name is not decided; it is the macro `\app` in `preamble.tex`.
- `build.sh` fails on LaTeX errors and undefined references.
