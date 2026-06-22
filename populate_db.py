#!/usr/bin/env python3
"""
Script to create a SQLite database from module JSON files.
Parses all JSON files under the `modules/` directory and populates
normalized tables covering every field.
"""

import json
import sqlite3
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_int(val):
    """Safely coerce a JSON value to int, returning None on failure / dash."""
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, (int, float)):
        return int(val)
    if isinstance(val, str):
        v = val.strip()
        if v in ("-", ""):
            return None
        try:
            return int(v)
        except ValueError:
            return None
    return None


def _to_float(val):
    """Safely coerce a JSON value to float, returning None on failure."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        v = val.strip()
        if v in ("-", ""):
            return None
        try:
            return float(v)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# Schema creation
# ---------------------------------------------------------------------------

def create_schema(conn: sqlite3.Connection) -> None:
    """Drop existing tables and recreate a normalized schema."""
    cursor = conn.cursor()
    cursor.executescript("""
        DROP TABLE IF EXISTS lernergebnisse_inhalte;
        DROP TABLE IF EXISTS lernergebnisse;
        DROP TABLE IF EXISTS lerninhalte_inhalte;
        DROP TABLE IF EXISTS lerninhalte;
        DROP TABLE IF EXISTS pruefungen;
        DROP TABLE IF EXISTS modulbestandteile;
        DROP TABLE IF EXISTS lehrveranstaltungen_art;
        DROP TABLE IF EXISTS lehrveranstaltungen;
        DROP TABLE IF EXISTS vergabe_kreditpunkte;
        DROP TABLE IF EXISTS formale_voraussetzungen;
        DROP TABLE IF EXISTS modules;

        CREATE TABLE modules (
            module_id           INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_id        TEXT,
            modulnummer         TEXT,
            name                TEXT,
            semester            INTEGER,
            modultyp            TEXT,
            credits             INTEGER,
            workload_stunden    INTEGER,
            verantwortlich      TEXT,
            haeufigkeit_des_angebots TEXT,
            dauer               TEXT,
            quelle_dokument     TEXT,
            quelle_seite_von    INTEGER,
            quelle_seite_bis    INTEGER
        );

        CREATE TABLE formale_voraussetzungen (
            module_id       INTEGER NOT NULL,
            voraussetzung   TEXT NOT NULL,
            PRIMARY KEY (module_id, voraussetzung),
            FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
        );

        CREATE TABLE vergabe_kreditpunkte (
            module_id   INTEGER NOT NULL,
            regel       TEXT NOT NULL,
            PRIMARY KEY (module_id, regel),
            FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
        );

        -- Sub-module type A: teaching events (lehrveranstaltungen)
        CREATE TABLE lehrveranstaltungen (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id               INTEGER NOT NULL,
            lv_id                   TEXT,
            lv_name                 TEXT,
            dozent                  TEXT,
            sws                     REAL,
            kontaktstunden          INTEGER,
            selbststudiumsstunden   INTEGER,
            FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
        );

        CREATE TABLE lehrveranstaltungen_art (
            lehrveranstaltung_id    INTEGER NOT NULL,
            art                     TEXT NOT NULL,
            PRIMARY KEY (lehrveranstaltung_id, art),
            FOREIGN KEY (lehrveranstaltung_id) REFERENCES lehrveranstaltungen(id) ON DELETE CASCADE
        );

        -- Sub-module type B: module components (modulbestandteile)
        CREATE TABLE modulbestandteile (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id               INTEGER NOT NULL,
            name                    TEXT,
            dozent                  TEXT,
            kontaktzeit             TEXT,
            selbststudiumsstunden   INTEGER,
            FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
        );

        CREATE TABLE pruefungen (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id           INTEGER NOT NULL,
            bezug               TEXT,
            form                TEXT,
            dauer_minuten       INTEGER,
            umfang_seiten       INTEGER,
            bearbeitungszeitraum TEXT,
            bearbeitungsdauer   TEXT,
            FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
        );

        CREATE TABLE lerninhalte (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id   INTEGER NOT NULL,
            bezug       TEXT,
            FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
        );

        CREATE TABLE lerninhalte_inhalte (
            lerninhalt_id   INTEGER NOT NULL,
            inhalt          TEXT NOT NULL,
            PRIMARY KEY (lerninhalt_id, inhalt),
            FOREIGN KEY (lerninhalt_id) REFERENCES lerninhalte(id) ON DELETE CASCADE
        );

        CREATE TABLE lernergebnisse (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            module_id   INTEGER NOT NULL,
            bezug       TEXT,
            FOREIGN KEY (module_id) REFERENCES modules(module_id) ON DELETE CASCADE
        );

        CREATE TABLE lernergebnisse_inhalte (
            lernergebnis_id INTEGER NOT NULL,
            inhalt          TEXT NOT NULL,
            PRIMARY KEY (lernergebnis_id, inhalt),
            FOREIGN KEY (lernergebnis_id) REFERENCES lernergebnisse(id) ON DELETE CASCADE
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Module processing
# ---------------------------------------------------------------------------

def process_module(conn: sqlite3.Connection, filepath: Path) -> None:
    """Read a single JSON module file and insert every field into the DB."""
    with open(filepath, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    cursor = conn.cursor()

    # --- quelle ---------------------------------------------------------------
    quelle = data.get("quelle", {})
    seiten = quelle.get("seiten", [])
    seite_von = seiten[0] if len(seiten) >= 1 else None
    seite_bis = seiten[1] if len(seiten) >= 2 else None

    # --- weitere_informationen ------------------------------------------------
    wi = data.get("weitere_informationen", {})
    if not isinstance(wi, dict):
        wi = {}

    reference_id = data.get("id")
    modul_name = data.get("name")

    cursor.execute(
        """
        INSERT INTO modules (
            reference_id, modulnummer, name, semester, modultyp, credits,
            workload_stunden, verantwortlich, haeufigkeit_des_angebots, dauer,
            quelle_dokument, quelle_seite_von, quelle_seite_bis
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reference_id,
            data.get("modulnummer"),
            modul_name,
            _to_int(data.get("semester")),
            data.get("modultyp"),
            _to_int(data.get("credits")),
            _to_int(data.get("workload_stunden")),
            data.get("verantwortlich"),
            wi.get("haeufigkeit_des_angebots"),
            wi.get("dauer"),
            quelle.get("dokument"),
            _to_int(seite_von),
            _to_int(seite_bis),
        ),
    )
    module_id = cursor.lastrowid

    # --- formale_voraussetzungen (string or list) -----------------------------
    formal = wi.get("formale_voraussetzungen")
    if formal is not None:
        entries = formal if isinstance(formal, list) else [formal]
        for v in entries:
            cursor.execute(
                """
                INSERT OR IGNORE INTO formale_voraussetzungen (module_id, voraussetzung)
                VALUES (?, ?)
                """,
                (module_id, v),
            )

    # --- vergabe_kreditpunkte -------------------------------------------------
    kredite = wi.get("vergabe_kreditpunkte")
    if isinstance(kredite, list):
        for regel in kredite:
            cursor.execute(
                """
                INSERT OR IGNORE INTO vergabe_kreditpunkte (module_id, regel)
                VALUES (?, ?)
                """,
                (module_id, regel),
            )

    # --- submodules A: lehrveranstaltungen ------------------------------------
    for lv in data.get("lehrveranstaltungen", []):
        cursor.execute(
            """
            INSERT INTO lehrveranstaltungen
                (module_id, lv_id, lv_name, dozent, sws, kontaktstunden, selbststudiumsstunden)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                module_id,
                lv.get("lv_id") or None,
                lv.get("name"),
                lv.get("dozent"),
                _to_float(lv.get("sws")),
                _to_int(lv.get("kontaktstunden")),
                _to_int(lv.get("selbststudiumsstunden")),
            ),
        )
        lv_db_id = cursor.lastrowid
        for art in lv.get("art", []):
            cursor.execute(
                """
                INSERT OR IGNORE INTO lehrveranstaltungen_art (lehrveranstaltung_id, art)
                VALUES (?, ?)
                """,
                (lv_db_id, art),
            )

    # --- submodules B: modulbestandteile --------------------------------------
    for mb in data.get("modulbestandteile", []):
        cursor.execute(
            """
            INSERT INTO modulbestandteile
                (module_id, name, dozent, kontaktzeit, selbststudiumsstunden)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                module_id,
                mb.get("name"),
                mb.get("dozent"),
                mb.get("kontaktzeit"),
                _to_int(mb.get("selbststudiumsstunden")),
            ),
        )

    # --- examinations ---------------------------------------------------------
    for p in data.get("pruefungen", []):
        cursor.execute(
            """
            INSERT INTO pruefungen
                (module_id, bezug, form, dauer_minuten, umfang_seiten, bearbeitungszeitraum, bearbeitungsdauer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                module_id,
                p.get("bezug"),
                p.get("form"),
                _to_int(p.get("dauer_minuten")),
                _to_int(p.get("umfang_seiten")),
                p.get("bearbeitungszeitraum"),
                p.get("bearbeitungsdauer"),
            ),
        )

    # --- learning contents ----------------------------------------------------
    for li in data.get("lerninhalte", []):
        cursor.execute(
            "INSERT INTO lerninhalte (module_id, bezug) VALUES (?, ?)",
            (module_id, li.get("bezug")),
        )
        li_db_id = cursor.lastrowid
        for inhalt in li.get("inhalte", []):
            cursor.execute(
                """
                INSERT OR IGNORE INTO lerninhalte_inhalte (lerninhalt_id, inhalt)
                VALUES (?, ?)
                """,
                (li_db_id, inhalt),
            )

    # --- learning outcomes / competencies -------------------------------------
    for lk in data.get("lernergebnisse_kompetenzen", []):
        cursor.execute(
            "INSERT INTO lernergebnisse (module_id, bezug) VALUES (?, ?)",
            (module_id, lk.get("bezug")),
        )
        lk_db_id = cursor.lastrowid
        for inhalt in lk.get("inhalte", []):
            cursor.execute(
                """
                INSERT OR IGNORE INTO lernergebnisse_inhalte (lernergebnis_id, inhalt)
                VALUES (?, ?)
                """,
                (lk_db_id, inhalt),
            )

    conn.commit()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    modules_dir = Path(__file__).parent / "modules"
    db_path = Path(__file__).parent / "study_buddy.db"

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    create_schema(conn)

    json_files = sorted(modules_dir.rglob("*.json"))
    if not json_files:
        print(f"No JSON files found in {modules_dir}")
        conn.close()
        return

    for filepath in json_files:
        print(f"Processing {filepath}")
        process_module(conn, filepath)

    # Quick sanity check
    cursor = conn.cursor()
    counts = {
        "modules": "SELECT COUNT(*) FROM modules",
        "formale_voraussetzungen": "SELECT COUNT(*) FROM formale_voraussetzungen",
        "vergabe_kreditpunkte": "SELECT COUNT(*) FROM vergabe_kreditpunkte",
        "lehrveranstaltungen": "SELECT COUNT(*) FROM lehrveranstaltungen",
        "lehrveranstaltungen_art": "SELECT COUNT(*) FROM lehrveranstaltungen_art",
        "modulbestandteile": "SELECT COUNT(*) FROM modulbestandteile",
        "pruefungen": "SELECT COUNT(*) FROM pruefungen",
        "lerninhalte": "SELECT COUNT(*) FROM lerninhalte",
        "lerninhalte_inhalte": "SELECT COUNT(*) FROM lerninhalte_inhalte",
        "lernergebnisse": "SELECT COUNT(*) FROM lernergebnisse",
        "lernergebnisse_inhalte": "SELECT COUNT(*) FROM lernergebnisse_inhalte",
    }
    print(f"\nDatabase created at {db_path}")
    for label, query in counts.items():
        cursor.execute(query)
        count = cursor.fetchone()[0]
        print(f"  {label:30s} : {count}")

    conn.close()


if __name__ == "__main__":
    main()
