# COMINT
Radio frequency and traffic analysis

This Python script is a example component of a broader research and development pipeline designed to extract structured intelligence from Software Defined Radio (SDR) transmissions. It integrates audio transcription (via OpenAI Whisper), frequency-to-agency contextual mapping, and geographic toponym verification using OpenStreetMap data. The script supports downstream tasks like determining the "Who, What, When, Where" (4W) from voice communications intercepted through SDR monitoring.

Features:

Automatic transcription of audio transmissions using Whisper (large-v2, CUDA-enabled).

Contextual metadata extraction from local and national frequency databases.

Toponym verification using the Overpass API against OpenStreetMap’s hierarchical regions.

Geospatial proximity analysis to associate ambiguous toponyms with relevant jurisdictions.

Pydantic models for validated, structured data handling.

Planned Integrations:

Real-time input from live SDR streams.

Direction-finding modules for triangulating transmission origins.

LLM-based post-processing of summaries in context with one another.

Technologies: Python · Whisper · OpenStreetMap API · Pandas · Geopy · Torch · Pydantic · SDR Metadata
