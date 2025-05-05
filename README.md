# COMINT

## Radio Frequency and Traffic Analysis

![KrakenSDR](https://github.com/stratas-data/COMINT/blob/main/IMG_3364.jpg)
(First run of a KrakenSDR)

The attached Python script is a example component of a broader research and development pipeline designed to extract structured intelligence from Software Defined Radio (SDR) transmissions. It integrates audio transcription (via OpenAI Whisper), frequency-to-agency contextual mapping, and geographic toponym verification using OpenStreetMap data. The script supports downstream tasks like determining the "Who, What, When, Where" (4W) from voice communications intercepted through SDR monitoring.

The following summary of a transmission was generated in approximately twenty seconds. In this example, OpenStreeMap's API and cascading logic was used, which delivered toponym disambiguations an analyst can use to quickly know the exact location of the event:
> who: nil,

> what: Issues with traffic light not functioning properly, will call for tech support.,

> when: Last 10 minutes,

> where:

>   toponym: Brookshire's,

>   verified: true,

>   note: Wise County is the nearest relevant county to 'Brookshire's' (33.2249552, -97.7547486), distance: 9.65 km.
   
>   toponym: Whataburger,

>   verified: true,

>   note: Wise County is the nearest relevant county to 'Whataburger' (33.2303451, -97.5956807), distance: 5.62 km.

### Features:

Automatic transcription of audio transmissions using Whisper (large-v2, CUDA-enabled).

Contextual metadata extraction from local and national frequency databases.

Toponym verification using the Overpass API against OpenStreetMap’s hierarchical regions.

Geospatial proximity analysis to associate ambiguous toponyms with relevant jurisdictions.

Pydantic models for validated, structured data handling.

### Planned Integrations:

Real-time input from live SDR streams.

Direction-finding modules for triangulating transmission origins.

LLM-based post-processing of summaries in context with one another.

### Technologies used:

Python · Whisper · OpenStreetMap API · Pandas · Geopy · Torch · Pydantic · SDR Metadata
