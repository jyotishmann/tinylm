## Summary

chore: initialise repository structure

## Why

Sets up the complete directory skeleton before any source code is written. Every subsequent PR has a pre-existing home for its files. `.gitignore` is established now so no accidental checkpoint or corpus commits can occur. README skeleton makes the repo page intentional from day one.

## Changes

- Creates full directory skeleton for all 8 phases
- Adds __init__.py to all Python packages
- Adds .gitkeep to preserve empty data/checkpoint/log dirs
- Adds .gitignore covering Python, PyTorch, Node, OS artefacts
- Adds README skeleton and GitHub PR template
 
## Testing

`find . -name "*.py" | grep __init__` should list all 7 package init files.

## Notes

<!-- Anything the reviewer should know: trade-offs made, alternatives considered, follow-up PRs needed -->

## References

<!-- MASTER.md section, ADR number, or external link -->