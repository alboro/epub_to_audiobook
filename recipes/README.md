# Recipes

This folder contains reusable launch recipes for specific deployment setups.

Principles:

- recipes are commit-friendly
- recipes should avoid hardcoding personal secrets or machine-specific paths
- local convenience wrappers can live outside the repo or under ignored folders such as `.local/`

Current recipe:

- `win_ru_xtts`
  A practical Russian XTTS workflow that targets a local or remote polling TTS server.
