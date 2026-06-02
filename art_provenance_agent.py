"""
Art Provenance Research Agent

Uses Claude + Tavily search to build a documented chain of ownership
for artworks by searching auction records, museum records, dealer records,
exhibition catalogs, and ownership transfers.

Requirements:
    pip install anthropic tavily-python
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import anthropic
from tavily import TavilyClient


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

SEARCH_TOOL = {
    "name": "search_provenance",
    "description": (
        "Search for art provenance records including auction results, museum collection history, "
        "dealer transactions, exhibition catalogs, and ownership transfers. "
        "Call this when you need documentation about an artwork's chain of custody. "
        "Make targeted queries — e.g. 'Monet Water Lilies 1906 Christie's auction' — "
        "rather than broad ones."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Specific search query. Include the artwork title and/or artist name "
                    "plus the record type you are looking for "
                    "(auction, exhibition, museum, restitution, etc.)."
                ),
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "description": (
                    "Use 'advanced' for thorough provenance research; "
                    "'basic' for quick cross-checks."
                ),
            },
        },
        "required": ["query"],
    },
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert art provenance researcher with deep knowledge of auction houses,
museum cataloguing standards, and legal requirements for establishing clear title to artworks.

Your task is to build a complete provenance log for a given artwork by conducting multiple targeted
Tavily searches. Research these six evidence categories in order:

1. **Auction records** — sale dates, hammer prices, auction house (Christie's, Sotheby's, Bonhams,
   Phillips, Dorotheum, etc.), lot numbers, catalogue entries.
2. **Museum records** — current/past institutional ownership, exhibition loans, accession numbers,
   deaccessions, bequest or donation records.
3. **Dealer records** — gallery sales, private dealer transactions, art fair appearances.
4. **Exhibition catalogs** — group or solo shows listing the work; catalogue essays are primary sources.
5. **Ownership transfers** — documented private sales, inheritance records, estate dispersals.
6. **Restitution / claims** — especially critical for European works created or traded 1933–1945;
   check the Art Loss Register, Commission for Looted Art, and national databases.

After gathering evidence, compile a structured provenance log:
- List each ownership period in chronological order.
- For each entry include: owner name/entity, acquisition date (or range), disposal date (or range),
  and the source document that establishes the link.
- Flag any provenance gaps (periods with no documented owner).
- Note any red flags: missing documentation, disputed attribution, wartime gaps, or open restitution claims.

Be thorough — run at least 4–6 searches per artwork before concluding."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceEntry:
    owner: str
    from_date: str          # YYYY or YYYY-MM or "unknown"
    to_date: str            # YYYY or YYYY-MM or "present" or "unknown"
    acquisition_method: str # auction / gift / purchase / bequest / inheritance / unknown
    source: str             # URL or document citation
    notes: str = ""


@dataclass
class ProvenanceLog:
    artwork_title: str
    artist: str
    approximate_date: str
    entries: list[ProvenanceEntry] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    red_flags: list[str] = field(default_factory=list)
    sources_consulted: list[str] = field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# Tavily tool execution
# ---------------------------------------------------------------------------

def execute_search(tavily: TavilyClient, tool_input: dict) -> str:
    """Run a Tavily search and return results as a JSON string for Claude."""
    query = tool_input["query"]
    depth = tool_input.get("search_depth", "advanced")

    result = tavily.search(
        query=query,
        search_depth=depth,
        include_answer=True,
        include_raw_content=False,
        max_results=6,
    )

    # Distill into a compact format Claude can reason over
    distilled = {
        "query": query,
        "answer": result.get("answer", ""),
        "results": [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:400],
                "score": round(r.get("score", 0), 3),
            }
            for r in result.get("results", [])
        ],
    }
    return json.dumps(distilled, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def research_provenance(
    title: str,
    artist: str,
    approximate_date: str,
    *,
    anthropic_api_key: Optional[str] = None,
    tavily_api_key: Optional[str] = None,
    max_iterations: int = 20,
) -> ProvenanceLog:
    """
    Run the provenance research agent and return a structured ProvenanceLog.

    Args:
        title:            Artwork title, e.g. "Water Lilies"
        artist:           Artist name, e.g. "Claude Monet"
        approximate_date: Creation date or range, e.g. "1906" or "c. 1900–1910"
        anthropic_api_key: Overrides ANTHROPIC_API_KEY env var.
        tavily_api_key:    Overrides TAVILY_API_KEY env var.
        max_iterations:   Safety cap on agentic loop iterations.
    """
    client = anthropic.Anthropic(api_key=anthropic_api_key or os.environ["ANTHROPIC_API_KEY"])
    tavily = TavilyClient(api_key=tavily_api_key or os.environ["TAVILY_API_KEY"])

    user_message = (
        f"Research the complete provenance of the following artwork:\n\n"
        f"**Title:** {title}\n"
        f"**Artist:** {artist}\n"
        f"**Approximate date:** {approximate_date}\n\n"
        "Use the search_provenance tool multiple times to find auction records, museum records, "
        "dealer records, exhibition catalogs, ownership transfers, and any restitution claims. "
        "After your research, return a JSON object with this exact schema:\n\n"
        "```json\n"
        "{\n"
        '  "artwork_title": "...",\n'
        '  "artist": "...",\n'
        '  "approximate_date": "...",\n'
        '  "entries": [\n'
        "    {\n"
        '      "owner": "...",\n'
        '      "from_date": "YYYY or unknown",\n'
        '      "to_date": "YYYY or present or unknown",\n'
        '      "acquisition_method": "auction|gift|purchase|bequest|inheritance|unknown",\n'
        '      "source": "URL or citation",\n'
        '      "notes": "optional context"\n'
        "    }\n"
        "  ],\n"
        '  "gaps": ["describe each period with no documented owner"],\n'
        '  "red_flags": ["list any concerns"],\n'
        '  "sources_consulted": ["list all URLs or documents searched"],\n'
        '  "summary": "2–3 sentence overview of provenance quality"\n'
        "}\n"
        "```"
    )

    messages: list[anthropic.types.MessageParam] = [
        {"role": "user", "content": user_message}
    ]

    print(f"\n{'='*60}")
    print(f"Researching provenance: {title} by {artist} ({approximate_date})")
    print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        response = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=8096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract the JSON provenance log from the final text response
            final_text = ""
            for block in response.content:
                if block.type == "text":
                    final_text += block.text

            return _parse_provenance_log(final_text, title, artist, approximate_date)

        if response.stop_reason != "tool_use":
            print(f"[warn] Unexpected stop_reason: {response.stop_reason}")
            break

        # Execute tool calls
        tool_results: list[anthropic.types.ToolResultBlockParam] = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  🔍 Searching: {block.input.get('query', '')}")
                result_content = execute_search(tavily, block.input)
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_content,
                    }
                )

        messages.append({"role": "user", "content": tool_results})

    # Fallback if loop exhausted
    print("[warn] Max iterations reached; returning partial log.")
    return ProvenanceLog(
        artwork_title=title,
        artist=artist,
        approximate_date=approximate_date,
        summary="Research incomplete — max iterations reached.",
    )


# ---------------------------------------------------------------------------
# JSON → dataclass parser
# ---------------------------------------------------------------------------

def _parse_provenance_log(
    text: str, title: str, artist: str, approximate_date: str
) -> ProvenanceLog:
    """Extract the JSON provenance log embedded in Claude's response."""
    # Find the JSON block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return ProvenanceLog(
            artwork_title=title,
            artist=artist,
            approximate_date=approximate_date,
            summary=text.strip(),
        )

    try:
        data = json.loads(text[start:end])
    except json.JSONDecodeError:
        return ProvenanceLog(
            artwork_title=title,
            artist=artist,
            approximate_date=approximate_date,
            summary=text.strip(),
        )

    entries = [
        ProvenanceEntry(
            owner=e.get("owner", "Unknown"),
            from_date=e.get("from_date", "unknown"),
            to_date=e.get("to_date", "unknown"),
            acquisition_method=e.get("acquisition_method", "unknown"),
            source=e.get("source", ""),
            notes=e.get("notes", ""),
        )
        for e in data.get("entries", [])
    ]

    return ProvenanceLog(
        artwork_title=data.get("artwork_title", title),
        artist=data.get("artist", artist),
        approximate_date=data.get("approximate_date", approximate_date),
        entries=entries,
        gaps=data.get("gaps", []),
        red_flags=data.get("red_flags", []),
        sources_consulted=data.get("sources_consulted", []),
        summary=data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_provenance_log(log: ProvenanceLog) -> None:
    """Print the provenance log to stdout in a readable format."""
    print(f"\n{'='*60}")
    print(f"PROVENANCE LOG")
    print(f"{'='*60}")
    print(f"Artwork : {log.artwork_title}")
    print(f"Artist  : {log.artist}")
    print(f"Date    : {log.approximate_date}")
    print(f"\nSUMMARY\n{'-'*40}")
    print(log.summary)

    if log.entries:
        print(f"\nCHAIN OF OWNERSHIP ({len(log.entries)} records)\n{'-'*40}")
        for i, entry in enumerate(log.entries, 1):
            print(f"{i}. [{entry.from_date} – {entry.to_date}]  {entry.owner}")
            print(f"   Method : {entry.acquisition_method}")
            print(f"   Source : {entry.source}")
            if entry.notes:
                print(f"   Notes  : {entry.notes}")

    if log.gaps:
        print(f"\nPROVENANCE GAPS\n{'-'*40}")
        for gap in log.gaps:
            print(f"  ⚠  {gap}")

    if log.red_flags:
        print(f"\nRED FLAGS\n{'-'*40}")
        for flag in log.red_flags:
            print(f"  🚩 {flag}")

    if log.sources_consulted:
        print(f"\nSOURCES CONSULTED ({len(log.sources_consulted)})\n{'-'*40}")
        for src in log.sources_consulted:
            print(f"  • {src}")

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Example artwork — override via command-line args: title, artist, date
    if len(sys.argv) == 4:
        artwork_title = sys.argv[1]
        artwork_artist = sys.argv[2]
        artwork_date = sys.argv[3]
    else:
        # Default demo artwork
        artwork_title = "Sunflowers"
        artwork_artist = "Vincent van Gogh"
        artwork_date = "1888"

    log = research_provenance(
        title=artwork_title,
        artist=artwork_artist,
        approximate_date=artwork_date,
    )

    print_provenance_log(log)

    # Optionally save JSON
    output_file = f"provenance_{artwork_artist.replace(' ', '_')}_{artwork_title.replace(' ', '_')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(asdict(log), f, indent=2, ensure_ascii=False)
    print(f"Full log saved to: {output_file}")
