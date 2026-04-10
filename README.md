# EPUB to Audiobook Converter

This repository is a fork of [p0n1/epub_to_audiobook](https://github.com/p0n1/epub_to_audiobook).

The original project converts EPUB ebooks into audiobooks with multiple TTS backends.

This fork keeps that base and adds work aimed at a more practical self-hosted workflow:

- optional LLM text normalizer before TTS
- optional polling-based TTS flow for OpenAI-compatible job APIs
- optional `m4b` packaging with `ffmpeg`

The goal of this fork is still simple:

`EPUB -> audiobook`

Default behavior is intended to remain close to the original project unless the new options are explicitly enabled.
