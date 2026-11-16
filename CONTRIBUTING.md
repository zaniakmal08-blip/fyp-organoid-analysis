# Personal development workflow

1. Open an issue for one research question, bug or milestone.
2. Create a short branch: `git switch -c feature/short-description`.
3. Make the change; update the relevant experiment/progress record.
4. Run applicable checks from `docs/SETUP.md`.
5. Review `git status` and `git diff` before staging named files. Avoid adding raw data or private material.
6. Commit a clear message, push the branch and open a pull request. Self-review and merge after checks pass.

Every supervisor update should distinguish completed work, evidence, limitations, decisions needed and next actions. Do not describe model fitting scores as test accuracy. Keep original data immutable; derived labels and corrected measurements need a separate provenance record.
