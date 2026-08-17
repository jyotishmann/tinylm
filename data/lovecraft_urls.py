# data/lovecraft_urls.py
# Curated list of H.P. Lovecraft works from Project Gutenberg.
# All texts are public domain (Lovecraft died 1937).
#
# IMPORTANT: Verify these URLs before running download.py.
# Project Gutenberg occasionally migrates file IDs.
# Each URL should serve a plain .txt file (not HTML).
# Verify at: https://www.gutenberg.org/ebooks/search/?query=lovecraft

LOVECRAFT_TEXTS = [
    # ── Major Mythos Stories ─────────────────────────────────────────
    {
        "title": "The Call of Cthulhu",
        "url": "https://www.gutenberg.org/cache/epub/68230/pg68230.txt",
        "pg_id": 68230,
    },
    {
        "title": "At the Mountains of Madness",
        "url": "https://www.gutenberg.org/cache/epub/70652/pg70652.txt",
        "pg_id": 70652,
    },
    {
        "title": "The Shadow Over Innsmouth",
        "url": "https://www.gutenberg.org/cache/epub/73181/pg73181.txt",
        "pg_id": 73181,
    },
    {
        "title": "The Dunwich Horror",
        "url": "https://www.gutenberg.org/cache/epub/50133/pg50133.txt",
        "pg_id": 50133,
    },
    # ── Other Major Works ────────────────────────────────────────────
    {
        "title": "The Case of Charles Dexter Ward",
        "url": "https://www.gutenberg.org/cache/epub/73547/pg73547.txt",
        "pg_id": 73547,
    },
    {
        "title": "The Colour Out of Space",
        "url": "https://www.gutenberg.org/cache/epub/68236/pg68236.txt",
        "pg_id": 68236,
    },
    {
        "title": "The Horror at Red Hook",
        "url": "https://www.gutenberg.org/cache/epub/72966/pg72966.txt",
        "pg_id": 72966,
    },
    {
        "title": "HE",
        "url": "https://www.gutenberg.org/cache/epub/68547/pg68547.txt",
        "pg_id": 68547,
    },    
    {
        "title": "The Festival",
        "url": "https://www.gutenberg.org/cache/epub/68553/pg68553.txt",
        "pg_id": 68553,
    },
    {
        "title": "The LURKING FEAR",
        "url": "https://www.gutenberg.org/cache/epub/70486/pg70486.txt",
        "pg_id": 70486,
    },
    {
        "title": "The Haunter of the Dark",
        "url": "https://www.gutenberg.org/cache/epub/73233/pg73233.txt",
        "pg_id": 73233,
    },
    {
        "title": "The Thing on the Door-Step",
        "url": "https://www.gutenberg.org/cache/epub/73230/pg73230.txt",
        "pg_id": 73230,
    },
    {
        "title": "The Quest of Iranon",
        "url": "https://www.gutenberg.org/cache/epub/73182/pg73182.txt",
        "pg_id": 73182,
    },

]

# Fallback: if individual URLs break, this omnibus compilation
# at gutenberg.org may be available. Check the PG search for
# "Lovecraft complete" to find the current omnibus ID.
OMNIBUS_FALLBACK = {
    "title": "The Complete Works of H.P. Lovecraft (fan compilation)",
    "url": "https://www.gutenberg.org/cache/epub/68283/pg68283.txt",
    "pg_id": 68283,
    "note": "Unofficial compilation — use individual stories if available",
}